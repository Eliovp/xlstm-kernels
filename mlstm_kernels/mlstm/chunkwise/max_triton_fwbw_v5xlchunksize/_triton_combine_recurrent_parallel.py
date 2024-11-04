# Copyright JKU Linz 2024
# Author: Maximilian Beck
"""This file contains the kernel that combines the recurrent and parallel part of the forward pass of the mLSTM chunkwise formulation.
It should allow arbitrary large chunk sizes and head dimensions.
"""

import torch

from ....kernel_utils import contiguous_noctx
from ._torch_chunkwise_gates import compute_chunkwise_log_gates_vecB_vecA
from ._triton_parallel_bw_dK import mlstm_chunkwise__parallel_bw_dK
from ._triton_parallel_bw_dQ import mlstm_chunkwise__parallel_bw_dQ
from ._triton_parallel_bw_dV import mlstm_chunkwise__parallel_bw_dV
from ._triton_parallel_fw import mlstm_chunkwise__parallel_fw_Hintra
from ._triton_recurrent_bw import mlstm_chunkwise__recurrent_bw_dC
from ._triton_recurrent_fw import mlstm_chunkwise__recurrent_fw_C

# TODO fuse the cumsum computations into the kernels, maybe?


@contiguous_noctx
def mlstm_chunkwise_fw(
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHV)
    vecI: torch.Tensor,  # (B, NH, S)
    vecF: torch.Tensor,  # (B, NH, S)
    matC_initial: torch.Tensor = None,  # (B, NH, DHQK, DHV)
    vecN_initial: torch.Tensor = None,  # (B, NH, DHQK)
    scaM_initial: torch.Tensor = None,  # (B, NH)
    qk_scale: float = None,
    return_last_states: bool = False,
    return_all_states: bool = False,
    chunk_size_inter: int = 64,
    chunk_size_intra: int = 64,
    siz_b_L_parallel: int = 32,
    siz_b_L_loop: int = 32,
    siz_b_DH_parallel: int | None = None,
    siz_b_DH_loop: int | None = None,
    num_warps_intra: int | None = None,
    num_warps_inter: int | None = None,
    num_stages_intra: int | None = None,
    num_stages_inter: int | None = None,
    output_dtype: torch.dtype = torch.float32,
    eps: float = 0.0,
) -> tuple[
    torch.Tensor,  # matH_out (B, NH, S, DHV)
    torch.Tensor,  # vecN_out (B, NH, S)
    torch.Tensor,  # vecM_out (B, NH, S)
    None
    | (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ),  # last_states (matC_states (B, NH, DHQK, DHV), vecN_states (B, NH, DHQK), scaMinter_states (B, NH))
    None
    | (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ),  # all_states (matC_states (B, NH, (NC+1) * DHQK, DHV), vecN_states (B, NH, (NC+1) * DHQK), scaMinter_states (B, NH, (NC+1)))
]:
    B, NH, S, DHQK = matQ.shape
    DHV = matV.shape[-1]
    assert (
        S % chunk_size_inter == 0
    ), f"Sequence length {S} is not divisible by inter chunk size {chunk_size_inter}."

    assert (
        S % chunk_size_intra == 0
    ), f"Sequence length {S} is not divisible by intra chunk size {chunk_size_intra}."

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    assert (
        chunk_size_inter <= chunk_size_intra
    ), f"chunk_size_inter {chunk_size_inter} must be >= chunk_size_intra {chunk_size_intra}"
    assert (
        chunk_size_intra % chunk_size_inter == 0
    ), f"chunk_size_intra {chunk_size_intra} must be divisible by chunk_size_inter {chunk_size_inter}"

    save_states_every_nth_chunk = chunk_size_intra // chunk_size_inter

    #! materialize the  C_k, n_k, m_k states for each chunk
    matC_k_states, vecN_k_states, scaMinter_k_states = mlstm_chunkwise__recurrent_fw_C(
        matK=matK,
        matV=matV,
        vecF=vecF,
        vecI=vecI,
        matC_initial=matC_initial,
        vecN_initial=vecN_initial,
        scaMinter_initial=scaM_initial,
        chunk_size=chunk_size_inter,
        save_states_every_nth_chunk=save_states_every_nth_chunk,
        num_stages=num_stages_inter,
        num_warps=num_warps_inter,
    )

    #! compute the outputs within each chunk
    matH_out, vecN_out, vecM_out = mlstm_chunkwise__parallel_fw_Hintra(
        matQ=matQ,
        matK=matK,
        matV=matV,
        vecI=vecI,
        vecF=vecF,
        matC_states=matC_k_states,
        vecN_states=vecN_k_states,
        scaMinter_states=scaMinter_k_states,
        qk_scale=qk_scale,
        chunk_size=chunk_size_intra,
        siz_b_LQ=siz_b_L_parallel,
        siz_b_LKV=siz_b_L_loop,
        siz_b_DHQK=siz_b_DH_loop,
        siz_b_DHHV=siz_b_DH_parallel,
        eps=eps,
        output_dtype=output_dtype,
        num_warps=num_warps_intra,
        num_stages=num_stages_intra,
    )

    ret_tuple = (
        matH_out,
        vecN_out,
        vecM_out,
    )
    if return_last_states:
        ret_tuple += (
            (
                matC_k_states[:, :, -DHQK:, :],
                vecN_k_states[:, :, -DHQK:],
                scaMinter_k_states[:, :, -1],
            ),
        )
    else:
        ret_tuple += (None,)

    if return_all_states:
        ret_tuple += ((matC_k_states, vecN_k_states, scaMinter_k_states),)
    else:
        ret_tuple += (None,)

    return ret_tuple  # (matH_out, vecN_out, vecM_out, optional(last_states), optional(all_states))


@contiguous_noctx
def mlstm_chunkwise_bw(
    ## Forward arguments
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHV)
    vecI: torch.Tensor,  # (B, NH, S)
    vecF: torch.Tensor,  # (B, NH, S)
    matC_initial: torch.Tensor = None,  # (B, NH, DHQK, DHV)
    vecN_initial: torch.Tensor = None,  # (B, NH, DHQK)
    scaM_initial: torch.Tensor = None,  # (B, NH)
    qk_scale: float = None,
    ## Backward arguments
    matCstate_all: torch.Tensor = None,  # (B, NH, (NCsaved+1) * DHQK, DHV)
    vecNstate_all: torch.Tensor = None,  # (B, NH, (NCsaved+1) * DHQK)
    scaMstate_all: torch.Tensor = None,  # (B, NH, (NCsaved+1))
    matH_out: torch.Tensor = None,  # (B, NH, S, DHV)
    vecN_out: torch.Tensor = None,  # (B, NH, S)
    vecM_out: torch.Tensor = None,  # (B, NH, S)
    matDeltaH_out: torch.Tensor = None,  # (B, NH, S, DHV)
    # vecDeltaN_out: torch.Tensor = None,  # (B, NH, S) # we probably do not want external gradients going into the denominator
    matDeltaC_last: torch.Tensor = None,  # (B, NH, DHQK, DHV)
    vecDeltaN_last: torch.Tensor = None,  # (B, NH, DHQK) # TODO not used, maybe leave out
    scaDeltaM_last: torch.Tensor = None,  # (B, NH) # TODO not used, maybe leave out
    ## Other arguments
    chunk_size_inter: int = 64,
    chunk_size_intra: int = 64,
    siz_b_L_parallel: int = 32,
    siz_b_L_loop: int = 32,
    siz_b_DH_parallel: int | None = None,
    siz_b_DH_loop: int | None = None,
    num_warps_intra: int | None = None,
    num_warps_inter: int | None = None,
    num_stages_intra: int | None = None,
    num_stages_inter: int | None = None,
    eps: float = 0.0,
    output_dtype: torch.dtype = torch.float32,
    compute_delta_n: bool = False,
):
    B, NH, S, DHQK = matQ.shape
    DHV = matV.shape[-1]

    assert (
        S % chunk_size_inter == 0
    ), f"Sequence length {S} is not divisible by chunk size inter {chunk_size_inter}."
    assert (
        S % chunk_size_intra == 0
    ), f"Sequence length {S} is not divisible by chunk size intra {chunk_size_intra}."

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    assert (
        chunk_size_inter <= chunk_size_intra
    ), f"chunk_size_inter {chunk_size_inter} must be >= chunk_size_intra {chunk_size_intra}"
    assert (
        chunk_size_intra % chunk_size_inter == 0
    ), f"chunk_size_intra {chunk_size_intra} must be divisible by chunk_size_inter {chunk_size_inter}"

    save_states_every_nth_chunk = chunk_size_intra // chunk_size_inter

    #! recompute the "all" states if needed
    if matCstate_all is None:
        assert (
            (matCstate_all is None)
            and (vecNstate_all is None)
            and (scaMstate_all is None)
        ), "Either all or none of the states must be provided."

        matCstate_all, vecNstate_all, scaMstate_all = mlstm_chunkwise__recurrent_fw_C(
            matK=matK,
            matV=matV,
            vecF=vecF,
            vecI=vecI,
            matC_initial=matC_initial,
            vecN_initial=vecN_initial,
            scaMinter_initial=scaM_initial,
            chunk_size=chunk_size_inter,
            save_states_every_nth_chunk=save_states_every_nth_chunk,
            num_stages=num_stages_inter,
            num_warps=num_warps_inter,
        )

    #! recurrent backward: compute the deltaC (& deltaN) gradients
    # matDeltaC_states (B, NH, (NC+1) * DHQK, DHHV)
    # TODO: return also matDeltaN_states (B, NH, (NC+1) * DHQK)
    matDeltaC_states = mlstm_chunkwise__recurrent_bw_dC(
        matQ=matQ,  # (B, NH, S, DHQK)
        vecF=vecF,  # (B, NH, S)
        scaM_inter=scaMstate_all,  # (B, NH, NCintra+1)
        vecM_combine=vecM_out,  # (B, NH, S)
        matDeltaH=matDeltaH_out,  # (B, NH, S, DHV)
        vecN_out=vecN_out,  # (B, NH, S)
        matH_out=matH_out,  # (B, NH, S, DHV)
        matDeltaC_last=matDeltaC_last,  # (B, NH, DHQK, DHV)
        qk_scale=qk_scale,
        chunk_size=chunk_size_inter,
        eps=eps,
        save_states_every_nth_chunk=save_states_every_nth_chunk,
        num_stages=num_stages_inter,
        num_warps=num_warps_inter,
    )

    #! parallel backward: compute the deltaQ, deltaK, deltaV gradients
    vecB, vecA = compute_chunkwise_log_gates_vecB_vecA(
        chunk_size=chunk_size_intra, vecI=vecI, vecF=vecF, return_vecB_only=False
    )

    #! compute deltaV
    matDeltaV = mlstm_chunkwise__parallel_bw_dV(
        matQ=matQ,
        matK=matK,
        matV=matV,  # unused
        vecI=vecI,
        vecB=vecB,
        vecA=vecA,
        matCstate_all=matCstate_all,  # unused
        vecNstate_all=vecNstate_all,  # unused
        scaMstate_all=scaMstate_all,
        matH_out=matH_out,  # unused
        vecN_out=vecN_out,
        vecM_out=vecM_out,
        matDeltaH_out=matDeltaH_out,
        matDeltaC_states=matDeltaC_states,
        vecDeltaN_states=None,  # unused # TODO compute this
        qk_scale=qk_scale,
        chunk_size=chunk_size_intra,
        siz_b_LQ=siz_b_L_loop,
        siz_b_LKV=siz_b_L_parallel,
        siz_b_DHQK=siz_b_DH_loop,
        siz_b_DHHV=siz_b_DH_parallel,
        num_warps=num_warps_intra,
        num_stages=num_stages_intra,
        eps=eps,
        output_dtype=matQ.dtype,
        compute_delta_n=False,  # TODO
    )

    #! compute deltaK
    matDeltaK = mlstm_chunkwise__parallel_bw_dK(
        matQ=matQ,
        matK=matK,
        matV=matV,  # unused
        vecI=vecI,
        vecB=vecB,
        vecA=vecA,
        matCstate_all=matCstate_all,  # unused
        vecNstate_all=vecNstate_all,  # unused
        scaMstate_all=scaMstate_all,
        matH_out=matH_out,  # unused
        vecN_out=vecN_out,
        vecM_out=vecM_out,
        matDeltaH_out=matDeltaH_out,
        matDeltaC_states=matDeltaC_states,
        vecDeltaN_states=None,  # unused # TODO compute this
        qk_scale=qk_scale,
        chunk_size=chunk_size_intra,
        siz_b_LQ=siz_b_L_loop,
        siz_b_LKV=siz_b_L_parallel,
        siz_b_DHQK=siz_b_DH_parallel,
        siz_b_DHHV=siz_b_DH_loop,
        num_warps=num_warps_intra,
        num_stages=num_stages_intra,
        eps=eps,
        output_dtype=matQ.dtype,
        compute_delta_n=False,  # TODO
    )

    #! compute deltaQ
    matDeltaQ = mlstm_chunkwise__parallel_bw_dQ(
        matQ=matQ,
        matK=matK,
        matV=matV,  # unused
        vecI=vecI,
        vecB=vecB,
        vecA=vecA,
        matCstate_all=matCstate_all,  # unused
        vecNstate_all=vecNstate_all,  # unused
        scaMstate_all=scaMstate_all,
        matH_out=matH_out,  # unused
        vecN_out=vecN_out,
        vecM_out=vecM_out,
        matDeltaH_out=matDeltaH_out,
        matDeltaC_states=matDeltaC_states,
        vecDeltaN_states=None,  # unused # TODO compute this
        qk_scale=qk_scale,
        chunk_size=chunk_size_intra,
        siz_b_LQ=siz_b_L_parallel,
        siz_b_LKV=siz_b_L_loop,
        siz_b_DHQK=siz_b_DH_parallel,
        siz_b_DHHV=siz_b_DH_loop,
        num_warps=num_warps_intra,
        num_stages=num_stages_intra,
        eps=eps,
        output_dtype=matQ.dtype,
        compute_delta_n=False,  # TODO
    )

    #! postprocessing: compute deltaF and deltaI gradients
    ## ? postprocessing
    # vecF = rearrange(vecF, "b nh nc l -> b nh (nc l)")
    # compute the vecDeltaFbar values with dfbar = rev_cumsum((q*dq - k*dk).sum(-1))
    matQ = matQ.to(torch.float32)
    matK = matK.to(torch.float32)
    matDeltaQ = matDeltaQ.to(torch.float32)
    matDeltaK = matDeltaK.to(torch.float32)
    vecDeltaFbar_acc = ((matQ * matDeltaQ) - (matK * matDeltaK)).sum(-1)
    vecDeltaFbar = vecDeltaFbar_acc.flip(-1).to(torch.float32).cumsum(-1).flip(-1)
    vecDeltaF = vecDeltaFbar * torch.sigmoid(-vecF)
    ## ? end postprocessing
    # compute deltaI
    # both are equivalent:
    # vecDeltaI = (matV * matDeltaV).sum(-1)
    vecDeltaI = (matK * matDeltaK).sum(-1)

    # vecDeltaI = torch.zeros((B, NH, S), dtype=vecI.dtype, device=vecI.device)

    matDeltaC_initial = (
        matDeltaC_states[:, :, :DHQK, :] if matC_initial is not None else None
    )
    # TODO compute this
    vecDeltaN_initial = (
        torch.zeros_like(vecN_initial) if vecN_initial is not None else None
    )
    scaDeltaM_initial = (
        torch.zeros_like(scaM_initial) if scaM_initial is not None else None
    )

    return (
        matDeltaQ,
        matDeltaK,
        matDeltaV,
        vecDeltaI,
        vecDeltaF,
        matDeltaC_initial,
        vecDeltaN_initial,
        scaDeltaM_initial,
    )
