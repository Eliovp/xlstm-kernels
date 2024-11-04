# Author: Maximilian Beck
"""This file contains the parallel part of the backward pass of the mLSTM chunkwise formulation,
i.e. the "intra-chunk" contribution that computes the deltaV gradients.

The work is partitioned such that there is no limit on either the chunk size or the qk or v dimension.
We use tiling in the chunk dimension L. We tile in Bq and Bkv blocks.
"""

import torch
import triton
import triton.language as tl

from ....kernel_utils import is_power_of_2, torch2triton_dtype


@triton.jit
def _mlstm_chunkwise_parallel_bw_dV_kernel(
    ## input tensor pointers
    matQ,  # (B, NH, S, DHQK)
    matK,  # (B, NH, S, DHQK)
    matV,  # (B, NH, S, DHHV)
    vecI,  # (B, NH, NC, L)
    vecB,  # (B, NH, NC, L)
    vecA,  # (B, NH, NC, L)
    matCstate_all,  # (B, NH, (NC+1) * DHQK, DHHV)
    vecNstate_all,  # (B, NH, (NC+1) * DHQK)
    scaMstate_all,  # (B, NH, (NC+1))
    matH_out,  # (B, NH, S, DHHV)
    vecN_out,  # (B, NH, S) # vecN_combine
    vecM_out,  # (B, NH, S) # vecM_combine
    matDeltaH_out,  # (B, NH, S, DHHV)
    matDeltaC_states,  # (B, NH, (NC+1) * DHQK, DHHV)
    vecDeltaN_states,  # (B, NH, (NC+1) * DHQK)
    ## output tensor pointers
    matDeltaV,  # (B, NH, S, DHHV)
    qk_scale,
    ## strides
    str_matQK_B_NH,
    str_matQK_S,
    str_matQK_DHQK,
    str_matHV_B_NH,
    str_matHV_S,
    str_matHV_DHHV,
    str_vecABI_B_NH,
    str_vecABI_NC,
    str_matCstate_B_NH,
    str_matCstate_NCDHQK,
    str_matCstate_DHHV,
    str_vecNstate_B_NH,
    str_scaMstate_B_NH,
    str_vecMN_B_NH,
    str_vecMN_S,
    ## dimensions
    B: tl.constexpr,
    NH: tl.constexpr,
    S: tl.constexpr,
    DHQK: tl.constexpr,
    DHHV: tl.constexpr,
    NC: tl.constexpr,
    L: tl.constexpr,
    ## block sizes
    siz_b_LQ: tl.constexpr,
    siz_b_LKV: tl.constexpr,
    siz_b_DHQK: tl.constexpr,
    siz_b_DHHV: tl.constexpr,
    ## other arguments
    DTYPE: tl.constexpr = tl.float32,
    OUTPUT_DTYPE: tl.constexpr = tl.float32,
    EPS: tl.constexpr = 0.0,
    COMPUTE_DELTA_N: tl.constexpr = False,
):
    # our grid has 4 dimensions: (num_b_DHHV, num_b_LKV, NC, B * NH)
    idx_b_DHHV, idx_b_LKV, idx_b_NC_BNH = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    idx_b_NC = idx_b_NC_BNH % NC
    idx_b_BNH = idx_b_NC_BNH // NC

    # gate pointers for the current thread block
    vecB_ptr = vecB + idx_b_BNH * str_vecABI_B_NH + idx_b_NC * str_vecABI_NC
    vecI_ptr = vecI + idx_b_BNH * str_vecABI_B_NH + idx_b_NC * str_vecABI_NC

    # load vecB_LKV (siz_b_LKV,)
    vecB_LKV_ptr = vecB_ptr + idx_b_LKV * siz_b_LKV + tl.arange(0, siz_b_LKV)
    vecB_LKV_val = tl.load(vecB_LKV_ptr).to(tl.float32)
    # load vecI_LKV (siz_b_LKV,)
    vecI_LKV_ptr = vecI_ptr + idx_b_LKV * siz_b_LKV + tl.arange(0, siz_b_LKV)
    vecI_LKV_val = tl.load(vecI_LKV_ptr).to(tl.float32)

    # ? compute vecAbar for inter chunk contribution
    # load scaM_val (1,)
    scaMinter_k_val = tl.load(scaMstate_all + idx_b_BNH * (NC + 1) + (idx_b_NC + 1)).to(
        tl.float32
    )
    # load vecA (siz_b_LKV,)
    vecA_ptr = (
        vecA
        + idx_b_BNH * str_vecABI_B_NH
        + idx_b_NC * str_vecABI_NC
        + idx_b_LKV * siz_b_LKV
        + tl.arange(0, siz_b_LKV)
    )
    vecA_val = tl.load(vecA_ptr).to(tl.float32)
    # compute vecAbar_val (siz_b_LKV,)
    vecAbar_val = tl.exp(vecA_val - scaMinter_k_val)

    # for causal masking
    b_kv_offset_start = idx_b_LKV * siz_b_LKV
    b_kv_offset_end = (idx_b_LKV + 1) * siz_b_LKV
    b_kv_idxes = b_kv_offset_start + tl.arange(0, siz_b_LKV)

    #! intra chunk contribution
    # init matDeltaV accumulator (siz_b_LKV, siz_b_DHHV)
    matDeltaV_acc = tl.zeros([siz_b_LKV, siz_b_DHHV], dtype=tl.float32)
    ##? loop over siz_b_LQ blocks
    # compute only upper triangular part of the matrix
    idx_b_LQ_start = (idx_b_LKV * siz_b_LKV) // siz_b_LQ
    idx_b_LQ_end = tl.cdiv(L, siz_b_LQ)
    for idx_b_LQ in range(idx_b_LQ_start, idx_b_LQ_end, 1):
        ## compute matS block (siz_b_LQ, siz_b_LKV) -> matS^T (siz_b_LKV, siz_b_LQ)
        ## init matS^T block accumulator (siz_b_LKV, siz_b_LQ)
        matS_trans_acc = tl.zeros([siz_b_LKV, siz_b_LQ], dtype=tl.float32)
        ###? loop over siz_b_DHQK blocks
        for idx_b_DHQK in range(tl.cdiv(DHQK, siz_b_DHQK)):
            ### load matK (non-transposed) (siz_b_LKV, siz_b_DHQK)
            matK_ptr = tl.make_block_ptr(
                base=matK + idx_b_BNH * str_matQK_B_NH,
                shape=(S, DHQK),
                strides=(str_matQK_S, str_matQK_DHQK),
                offsets=(idx_b_NC * L + idx_b_LKV * siz_b_LKV, idx_b_DHQK * siz_b_DHQK),
                block_shape=(siz_b_LKV, siz_b_DHQK),
                order=(1, 0),
            )
            matK_val = tl.load(matK_ptr, boundary_check=(0, 1)).to(tl.float32)

            #! inter chunk contribution
            # compute this only on the first iteration
            if idx_b_LQ == idx_b_LQ_start:
                # compute matKbar (siz_b_LKV, siz_b_DHQK)
                matKbar_val = matK_val * vecAbar_val[:, None]

                # load matDeltaC (siz_b_DHQK, siz_b_DHHV)
                # (idx_b_NC + 1) since matDeltaC_states contains all state delta errors also for the initial state (i.e. NC+1)
                # and in this kernel we take only the last NC states (we do not consider the initial state delta error)
                matDeltaC_ptr = tl.make_block_ptr(
                    base=matDeltaC_states
                    + idx_b_BNH * str_matCstate_B_NH
                    + (idx_b_NC + 1) * DHQK * DHHV,
                    shape=(DHQK, DHHV),
                    strides=(str_matCstate_NCDHQK, str_matCstate_DHHV),
                    offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_DHHV * siz_b_DHHV),
                    block_shape=(siz_b_DHQK, siz_b_DHHV),
                    order=(1, 0),
                )
                matDeltaC_val = tl.load(matDeltaC_ptr, boundary_check=(0, 1)).to(DTYPE)

                # compute matDeltaV_inter (siz_b_LKV, siz_b_DHHV)
                matDeltaV_acc += tl.dot(matKbar_val.to(DTYPE), matDeltaC_val)

            ### load matQ_trans (transposed) (siz_b_DHQK, siz_b_LQ)
            matQ_trans_ptr = tl.make_block_ptr(
                base=matQ + idx_b_BNH * str_matQK_B_NH,
                shape=(DHQK, S),
                strides=(str_matQK_DHQK, str_matQK_S),
                offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_NC * L + idx_b_LQ * siz_b_LQ),
                block_shape=(siz_b_DHQK, siz_b_LQ),
                order=(0, 1),
            )
            matQ_trans_val = tl.load(matQ_trans_ptr, boundary_check=(0, 1)).to(DTYPE)
            ### compute matS^T (siz_b_LKV, siz_b_LQ)
            matS_trans_acc += tl.dot(matK_val.to(DTYPE), matQ_trans_val)

        ###? end siz_b_DHQK loop

        # load vecB_LQ (siz_b_LQ,)
        vecB_LQ_ptr = vecB_ptr + idx_b_LQ * siz_b_LQ + tl.arange(0, siz_b_LQ)
        vecB_LQ_val = tl.load(vecB_LQ_ptr).to(tl.float32)

        ## compute matD block (siz_b_LQ, siz_b_LKV) -> matD^T (siz_b_LKV, siz_b_LQ)
        # construct gate matrix matDtilde (siz_b_LQ, siz_b_LKV)
        matDtilde_val = (
            vecB_LQ_val[:, None] - vecB_LKV_val[None, :] + vecI_LKV_val[None, :]
        )

        b_q_offset = idx_b_LQ * siz_b_LQ
        # causal masking if on the diagonal
        if b_kv_offset_end >= b_q_offset:
            b_q_idxes = b_q_offset + tl.arange(0, siz_b_LQ)
            mask = b_q_idxes[:, None] >= b_kv_idxes[None, :]
            matDtilde_val = tl.where(mask, matDtilde_val, -float("inf"))

        # load vecM_out (siz_b_LQ,)
        vecM_out_ptr = (
            vecM_out
            + idx_b_BNH * str_vecMN_B_NH
            + idx_b_NC * L
            + idx_b_LQ * siz_b_LQ
            + tl.arange(0, siz_b_LQ)
        )
        vecM_out_val = tl.load(vecM_out_ptr).to(tl.float32)

        # compute matD^T (siz_b_LKV, siz_b_LQ)
        matD_trans_val = tl.trans(tl.exp(matDtilde_val - vecM_out_val[:, None]))

        # compute matSbar^T (siz_b_LKV, siz_b_LQ)
        matSbar_trans_val = matS_trans_acc * qk_scale * matD_trans_val

        # load matDeltaH_out (siz_b_LQ, siz_b_DHHV)
        matDeltaH_out_ptr = tl.make_block_ptr(
            base=matDeltaH_out + idx_b_BNH * str_matHV_B_NH,
            shape=(S, DHHV),
            strides=(str_matHV_S, str_matHV_DHHV),
            offsets=(idx_b_NC * L + idx_b_LQ * siz_b_LQ, idx_b_DHHV * siz_b_DHHV),
            block_shape=(siz_b_LQ, siz_b_DHHV),
            order=(1, 0),
        )
        matDeltaH_out_val = tl.load(
            matDeltaH_out_ptr,
            boundary_check=(0, 1),
        ).to(tl.float32)

        # load vecN_out (siz_b_LQ,)
        vecN_out_ptr = (
            vecN_out
            + idx_b_BNH * str_vecMN_B_NH
            + idx_b_NC * L
            + idx_b_LQ * siz_b_LQ
            + tl.arange(0, siz_b_LQ)
        )
        vecN_out_val = tl.load(vecN_out_ptr).to(tl.float32)

        # compute matDeltaH_intra (siz_b_LQ, siz_b_DHHV)
        matDeltaH_intra_val = matDeltaH_out_val / (vecN_out_val[:, None] + EPS)

        ## accumulate matDeltaV (siz_b_LKV, siz_b_DHHV)
        matDeltaV_acc += tl.dot(
            matSbar_trans_val.to(DTYPE), matDeltaH_intra_val.to(DTYPE)
        )
        ##? end siz_b_LQ loop

    # store matDeltaV (siz_b_LKV, siz_b_DHHV)
    matDeltaV_ptr = tl.make_block_ptr(
        base=matDeltaV + idx_b_BNH * str_matHV_B_NH,
        shape=(S, DHHV),
        strides=(str_matHV_S, str_matHV_DHHV),
        offsets=(idx_b_NC * L + idx_b_LKV * siz_b_LKV, idx_b_DHHV * siz_b_DHHV),
        block_shape=(siz_b_LKV, siz_b_DHHV),
        order=(1, 0),
    )
    tl.store(matDeltaV_ptr, matDeltaV_acc.to(OUTPUT_DTYPE), boundary_check=(0, 1))


def mlstm_chunkwise__parallel_bw_dV(
    ## Forward arguments
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHHV)
    vecI: torch.Tensor,  # (B, NH, NC, L)
    vecA: torch.Tensor,  # (B, NH, NC, L)
    vecB: torch.Tensor,  # (B, NH, NC, L)
    ## Backward arguments
    matCstate_all: torch.Tensor,  # (B, NH, (NC+1) * DHQK, DHHV)
    vecNstate_all: torch.Tensor,  # (B, NH, (NC+1) * DHQK)
    scaMstate_all: torch.Tensor,  # (B, NH, (NC+1))
    matH_out: torch.Tensor,  # (B, NH, S, DHHV)
    vecN_out: torch.Tensor,  # (B, NH, S) # vecN_combine
    vecM_out: torch.Tensor,  # (B, NH, S) # vecM_combine
    matDeltaH_out: torch.Tensor,  # (B, NH, S, DHHV)
    # vecDeltaN_out: torch.Tensor = None,  # (B, NH, S) # we probably do not want external gradients going into the denominator
    matDeltaC_states: torch.Tensor,  # (B, NH, (NC+1) * DHQK, DHHV)
    vecDeltaN_states: torch.Tensor,  # (B, NH, (NC+1) * DHQK) # TODO: use this
    ## Other arguments
    qk_scale: float = None,
    chunk_size: int = 64,
    siz_b_LQ: int = 32,
    siz_b_LKV: int = 32,
    siz_b_DHQK: int | None = None,
    siz_b_DHHV: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    eps: float = 0.0,
    output_dtype: torch.dtype = torch.float32,
    compute_delta_n: bool = False,
) -> torch.Tensor:  # matDeltaV (B, NH, S, DHHV)
    """This function defines the grid and block sizes for the kernel launch and calls the kernel.
    chunk parallel size:        siz_b_LKV
    chunk loop size:            siz_b_LQ
    head dim parallel size:     siz_b_DHHV
    head dim loop size:         siz_b_DHQK
    """
    B, NH, S, DHQK = matQ.shape
    DHHV = matV.shape[-1]

    assert (
        S % chunk_size == 0
    ), f"Sequence length {S} must be divisible by chunk size {chunk_size}"
    NC = S // chunk_size
    L = chunk_size

    assert is_power_of_2(L), "Chunk size must be a power of 2."

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    siz_b_DHQK = (
        min(64, triton.next_power_of_2(DHQK)) if siz_b_DHQK is None else siz_b_DHQK
    )
    siz_b_DHHV = (
        min(128, triton.next_power_of_2(DHHV)) if siz_b_DHHV is None else siz_b_DHHV
    )

    assert siz_b_LQ <= L, "siz_b_LQ must be less than or equal to chunk size L"
    assert siz_b_LKV <= L, "siz_b_LKV must be less than or equal to chunk size L"
    assert siz_b_LQ <= siz_b_LKV, "siz_b_LQ must be less than or equal to siz_b_LKV"
    assert siz_b_LKV % siz_b_LQ == 0, "siz_b_LKV must be divisible by siz_b_LQ"

    num_b_DHHV = triton.cdiv(DHHV, siz_b_DHHV)
    num_b_LKV = triton.cdiv(L, siz_b_LKV)

    num_stages = 1 if num_stages is None else num_stages
    if num_warps is None:
        num_warps = 4 if (siz_b_DHQK >= 64 or siz_b_DHHV >= 64) else 2

    matDeltaV = torch.empty(B, NH, S, DHHV, device=matQ.device, dtype=output_dtype)
    grid = (num_b_DHHV, num_b_LKV, NC * B * NH)

    _mlstm_chunkwise_parallel_bw_dV_kernel[grid](
        matQ=matQ,
        matK=matK,
        matV=matV,
        vecI=vecI,
        vecB=vecB,
        vecA=vecA,
        matCstate_all=matCstate_all,
        vecNstate_all=vecNstate_all,
        scaMstate_all=scaMstate_all,
        matH_out=matH_out,
        vecN_out=vecN_out,
        vecM_out=vecM_out,
        matDeltaH_out=matDeltaH_out,
        matDeltaC_states=matDeltaC_states,
        vecDeltaN_states=vecDeltaN_states,
        matDeltaV=matDeltaV,
        qk_scale=qk_scale,
        str_matQK_B_NH=matQ.stride(1),
        str_matQK_S=matQ.stride(2),
        str_matQK_DHQK=matQ.stride(3),
        str_matHV_B_NH=matV.stride(1),
        str_matHV_S=matV.stride(2),
        str_matHV_DHHV=matV.stride(3),
        str_vecABI_B_NH=vecB.stride(1),
        str_vecABI_NC=vecB.stride(2),
        str_matCstate_B_NH=matCstate_all.stride(1),
        str_matCstate_NCDHQK=matCstate_all.stride(2),
        str_matCstate_DHHV=matCstate_all.stride(3),
        str_vecNstate_B_NH=vecNstate_all.stride(1),
        str_scaMstate_B_NH=scaMstate_all.stride(1),
        str_vecMN_B_NH=vecN_out.stride(1),
        str_vecMN_S=vecN_out.stride(2),
        B=B,
        NH=NH,
        S=S,
        DHQK=DHQK,
        DHHV=DHHV,
        NC=NC,
        L=L,
        siz_b_LQ=siz_b_LQ,
        siz_b_LKV=siz_b_LKV,
        siz_b_DHQK=siz_b_DHQK,
        siz_b_DHHV=siz_b_DHHV,
        DTYPE=torch2triton_dtype(matQ.dtype),
        OUTPUT_DTYPE=torch2triton_dtype(output_dtype),
        EPS=eps,
        COMPUTE_DELTA_N=compute_delta_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return matDeltaV
