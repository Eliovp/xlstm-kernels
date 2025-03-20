#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# This is a specialized version of the forward xLSTM kernel specifically optimized
# for head dimension 64 on AMD GPUs, which showed performance issues in benchmark tests.

import triton
import triton.language as tl

try:
    from ...amd_optimizations import is_amd, get_amd_kernel_config, is_cdna3
    AMD_SUPPORT = True
except ImportError:
    AMD_SUPPORT = False
    
    def is_amd():
        return False
    
    def get_amd_kernel_config(head_dim_q, head_dim_v=None):
        return {}
    
    def is_cdna3():
        return False


@triton.jit
def mlstm_chunkwise__parallel_fw_Hintra_kernel_amd_dim64(
    matQ,  # (B, NH, S, DHQK)
    matK,  # (B, NH, S, DHQK)
    matV,  # (B, NH, S, DHHV)
    # these are all the states at every chunk, (we only use NC states up to the last chunk, i.e. :-1)
    matC_states,  # (B, NH, (NC+1) * DHQK, DHHV)
    vecN_states,  # (B, NH, (NC+1) * DHQK)
    scaMinter_states,  # (B, NH, (NC+1))
    vecI,  # (B, NH, NC, L)
    vecB,  # (B, NH, NC, L)
    matHout,  # (B, NH, S, DHHV)
    vecNout,  # (B, NH, S)
    vecMout,  # (B, NH, S)
    qk_scale: tl.constexpr,
    str_matQK_B_NH: tl.constexpr,
    str_matQK_S: tl.constexpr,
    str_matQK_DHQK: tl.constexpr,
    str_matHV_B_NH: tl.constexpr,
    str_matHV_S: tl.constexpr,
    str_matHV_DHHV: tl.constexpr,
    str_matCstates_B_NH: tl.constexpr,
    str_matCstates_NCDHQK: tl.constexpr,
    str_matCstates_DHHV: tl.constexpr,
    str_vecNstates_B_NH: tl.constexpr,
    str_vecNstates_NCDHQK: tl.constexpr,
    str_scaMinterstates_B_NH: tl.constexpr,
    str_vecBI_B_NH: tl.constexpr,
    str_vecBI_NC: tl.constexpr,
    str_vecBI_L: tl.constexpr,
    str_vecMN_B_NH: tl.constexpr,
    str_vecMN_S: tl.constexpr,
    B: tl.constexpr,
    NH: tl.constexpr,
    S: tl.constexpr,
    DHQK: tl.constexpr,
    DHHV: tl.constexpr,
    NC: tl.constexpr,
    L: tl.constexpr,
    siz_b_LQ: tl.constexpr,
    siz_b_LKV: tl.constexpr,
    siz_b_DHQK: tl.constexpr,
    siz_b_DHHV: tl.constexpr,
    DTYPE: tl.constexpr = tl.float32,
    OUTPUT_DTYPE: tl.constexpr = tl.float32,
    EPS: tl.constexpr = 0.0,
    MINIMUM_MAX_VAL: tl.constexpr = -10.0,
):
    # Special kernel for head dimension 64 on AMD hardware
    
    # Our grid has more dimensions to increase parallelism:
    # (num_b_DHHV * 2, num_b_LQ * 2, (NC, B * NH))
    idx_b_DHHV, idx_b_LQ, idx_b_NC_BNH = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    
    # Compute chunk index and batch-head index
    idx_b_NC = idx_b_NC_BNH % NC
    idx_b_BNH = idx_b_NC_BNH // NC
    
    # For head_dim=64, we use smaller block sizes to avoid underutilization
    # Do bounds checking for all indices
    if (idx_b_DHHV * siz_b_DHHV >= DHHV or 
        idx_b_LQ * siz_b_LQ >= L or 
        idx_b_NC >= NC or 
        idx_b_BNH >= B * NH):
        return
    
    # Setup pointers for B vectors
    vecB_ptr = (
        vecB
        + idx_b_BNH * str_vecBI_B_NH
        + idx_b_NC * str_vecBI_NC
    )
    
    # Setup pointers for I vectors
    vecI_ptr = (
        vecI
        + idx_b_BNH * str_vecBI_B_NH
        + idx_b_NC * str_vecBI_NC
    )
    
    # Load scaMinter_km1 scalar
    scaMinter_km1 = tl.load(
        scaMinter_states
        + idx_b_BNH * str_scaMinterstates_B_NH
        + idx_b_NC
    ).to(tl.float32)
    
    # For smaller blocks, direct access to avoid shared memory overhead
    idx_b_LKV_start = 0
    idx_b_LKV_end = (L + siz_b_LKV - 1) // siz_b_LKV
    
    # We'll compute matH_intra_acc (siz_b_LQ, siz_b_DHHV)
    matH_intra_acc = tl.zeros([siz_b_LQ, siz_b_DHHV], dtype=tl.float32)
    # We'll compute vecN_intra_acc and vecM_intra_max (siz_b_LQ,)
    vecN_intra_acc = tl.zeros([siz_b_LQ], dtype=tl.float32)
    vecM_intra_max = tl.full([siz_b_LQ], MINIMUM_MAX_VAL, dtype=tl.float32)
    
    for idx_b_LKV in range(idx_b_LKV_start, idx_b_LKV_end):
        # compute matG = matQ @ matK^T
        # matG accumulator (siz_b_LQ, siz_b_LKV)
        matG = tl.zeros([siz_b_LQ, siz_b_LKV], dtype=tl.float32)
        
        # For head_dim=64, loop in smaller chunks to better utilize AMD's cache hierarchy
        for idx_b_DHQK in range(0, DHQK, 16):
            # load matQ block (siz_b_LQ, 16)
            # Special offset calculations to improve cache utilization
            q_offset_x = idx_b_NC * L + idx_b_LQ * siz_b_LQ
            q_offset_y = idx_b_DHQK
            
            # Direct loads with smaller block sizes for head_dim=64
            matQ_ptr = matQ + idx_b_BNH * str_matQK_B_NH
            matQ_ptr += q_offset_x * str_matQK_S + q_offset_y * str_matQK_DHQK
            
            # Use explicit load ranges to avoid boundary checks
            matQ_val = tl.load(
                matQ_ptr + tl.arange(0, siz_b_LQ)[:, None] * str_matQK_S + 
                tl.arange(0, min(16, DHQK - idx_b_DHQK))[None, :] * str_matQK_DHQK
            ).to(DTYPE)
            
            # load matK transposed block (16, siz_b_LKV)
            k_offset_x = idx_b_DHQK
            k_offset_y = idx_b_NC * L + idx_b_LKV * siz_b_LKV
            
            matK_ptr = matK + idx_b_BNH * str_matQK_B_NH
            matK_ptr += k_offset_x * str_matQK_DHQK + k_offset_y * str_matQK_S
            
            matK_val = tl.load(
                matK_ptr + tl.arange(0, min(16, DHQK - idx_b_DHQK))[:, None] * str_matQK_DHQK +
                tl.arange(0, siz_b_LKV)[None, :] * str_matQK_S
            ).to(DTYPE)
            
            # For head_dim=64, use direct matrix multiply for 16×16 blocks
            # which aligns better with AMD's hardware
            matG += tl.dot(matQ_val, matK_val)
            
        # load vecB_LKV (siz_B_LKV,)
        vecB_offset = idx_b_LKV * siz_b_LKV
        vecB_val = tl.load(
            vecB_ptr + vecB_offset + tl.arange(0, min(siz_b_LKV, L - vecB_offset))
        ).to(tl.float32)
        
        # load vecI_LKV (siz_B_LKV,)
        vecI_offset = idx_b_LKV * siz_b_LKV
        vecI_val = tl.load(
            vecI_ptr + vecI_offset + tl.arange(0, min(siz_b_LKV, L - vecI_offset))
        ).to(tl.float32)
        
        # process matG (siz_b_LQ, siz_b_LKV) to get matG (siz_b_LQ, siz_b_LKV)
        matG = matG * qk_scale
        
        # compute vecM_intra (siz_b_LQ, siz_b_LKV)
        vecM_intra = matG + vecB_val[None, :]
        
        # update vecM_intra_max (siz_b_LQ,) with row-wise max of vecM_intra
        vecM_intra_max_cand = tl.max(vecM_intra, 1)  # (siz_b_LQ,)
        vecM_intra_max = tl.maximum(vecM_intra_max, vecM_intra_max_cand)  # (siz_b_LQ,)
        
        # compute vecP_intra = e^(vecM_intra - vecM_intra_max) (siz_b_LQ, siz_b_LKV)
        vecP_intra = tl.exp(vecM_intra - vecM_intra_max[:, None])
        
        # compute vecN_intra (siz_b_LQ, siz_b_LKV)
        vecN_intra = vecP_intra * vecI_val[None, :]
        
        # load matV block (siz_b_LKV, siz_b_DHHV)
        # Use small blocks and loop for better cache utilization
        for idx_b_DHHV_inner in range(0, siz_b_DHHV, 16):
            v_offset_x = idx_b_NC * L + idx_b_LKV * siz_b_LKV
            v_offset_y = idx_b_DHHV * siz_b_DHHV + idx_b_DHHV_inner
            
            # Adjust for bounds
            inner_size = min(16, siz_b_DHHV - idx_b_DHHV_inner)
            
            matV_ptr = matV + idx_b_BNH * str_matHV_B_NH
            matV_ptr += v_offset_x * str_matHV_S + v_offset_y * str_matHV_DHHV
            
            matV_val = tl.load(
                matV_ptr + tl.arange(0, siz_b_LKV)[:, None] * str_matHV_S +
                tl.arange(0, inner_size)[None, :] * str_matHV_DHHV
            ).to(DTYPE)
            
            # Accumulate in small chunks for better cache efficiency
            matH_intra_update = tl.dot(vecN_intra, matV_val)
            matH_intra_acc[:, idx_b_DHHV_inner:idx_b_DHHV_inner+inner_size] += matH_intra_update
        
        # update vecN_intra_acc (siz_b_LQ,)
        vecN_intra_acc += tl.sum(vecN_intra, 1)  # (siz_b_LQ,)
    
    # compute vecM_combine_val (siz_b_LQ,)
    vecM_inter_val = scaMinter_km1  # scalar
    vecM_combine_val = tl.maximum(vecM_intra_max, vecM_inter_val)  # (siz_b_LQ,)
    
    # compute vecM_new_val (siz_b_LQ,)
    vecM_new_val = vecM_intra_max
    
    # compute the intra chunk attention contrib
    # compute the vecN_ratio_intra (siz_b_LQ,)
    vecN_ratio_intra = tl.exp(vecM_intra_max - vecM_combine_val)  # (siz_b_LQ,)
    
    # compute matH_intra_acc (siz_b_LQ, siz_b_DHHV)
    matH_intra_acc = matH_intra_acc / (vecN_intra_acc[:, None] + EPS)  # (siz_b_LQ, siz_b_DHHV)
    
    # compute vecN_intra_acc (siz_b_LQ,)
    vecN_intra_acc = vecN_intra_acc * vecN_ratio_intra  # (siz_b_LQ,)
    
    # ? compute the inter chunk attention contrib
    
    # compute the ratio for the maximum for the inter chunk Minter contribution
    vecM_ratio_inter = tl.exp(vecM_inter_val - vecM_combine_val)  # (siz_b_LQ,)
    
    # compute matH_inter (siz_b_LQ, siz_b_DHHV)
    matH_inter_acc = tl.zeros([siz_b_LQ, siz_b_DHHV], dtype=tl.float32)
    # compute vecN_inter (siz_b_LQ,)
    vecN_inter_acc = tl.zeros([siz_b_LQ], dtype=tl.float32)
    
    # Optimized for head_dim=64: process in smaller blocks
    for idx_b_DHQK_chunk in range(0, DHQK, 16):
        chunk_size = min(16, DHQK - idx_b_DHQK_chunk)
        
        # load matQ block (siz_b_LQ, 16)
        q_offset_x = idx_b_NC * L + idx_b_LQ * siz_b_LQ
        q_offset_y = idx_b_DHQK_chunk
        
        matQ_ptr = matQ + idx_b_BNH * str_matQK_B_NH
        matQ_ptr += q_offset_x * str_matQK_S + q_offset_y * str_matQK_DHQK
        
        matQ_val = tl.load(
            matQ_ptr + tl.arange(0, siz_b_LQ)[:, None] * str_matQK_S + 
            tl.arange(0, chunk_size)[None, :] * str_matQK_DHQK
        ).to(tl.float32)
        
        # Set up 1D array for vecBbar
        vecBbar_val = tl.full([siz_b_LQ], vecM_ratio_inter, dtype=tl.float32)
        matQbar_val = (matQ_val * vecBbar_val[:, None] * qk_scale).to(DTYPE)
        
        # Load and process in smaller chunks for matC_km1
        for idx_b_DHHV_inner in range(0, siz_b_DHHV, 16):
            inner_size = min(16, siz_b_DHHV - idx_b_DHHV_inner)
            
            # load matC_km1 (16, inner_size)
            matC_km1_ptr = (
                matC_states + idx_b_BNH * str_matCstates_B_NH + 
                idx_b_NC * DHQK * str_matCstates_NCDHQK
            )
            matC_km1_ptr += (idx_b_DHQK_chunk * str_matCstates_NCDHQK + 
                            (idx_b_DHHV * siz_b_DHHV + idx_b_DHHV_inner) * str_matCstates_DHHV)
            
            matC_km1_val = tl.load(
                matC_km1_ptr + tl.arange(0, chunk_size)[:, None] * str_matCstates_NCDHQK +
                tl.arange(0, inner_size)[None, :] * str_matCstates_DHHV
            ).to(DTYPE)
            
            # Accumulate in small blocks
            matH_inter_chunk = tl.dot(matQbar_val, matC_km1_val)
            matH_inter_acc[:, idx_b_DHHV_inner:idx_b_DHHV_inner+inner_size] += matH_inter_chunk
        
        # load vecN_km1 (16,)
        vecN_km1_ptr = (
            vecN_states + idx_b_BNH * str_vecNstates_B_NH + 
            idx_b_NC * DHQK + idx_b_DHQK_chunk
        )
        
        vecN_km1_val = tl.load(
            vecN_km1_ptr + tl.arange(0, chunk_size)
        ).to(tl.float32)
        
        # accumulate vecN_inter_acc (siz_b_LQ,)
        vecN_inter_acc += tl.sum(matQbar_val * vecN_km1_val[None, :], axis=1)
    
    # Combine intra and inter chunk contributions with vectorized operations
    vecM_comb_ratio = tl.exp(vecM_new_val - vecM_combine_val)
    
    # Efficient vector operations
    matH_comb_num_val = matH_inter_acc + vecM_comb_ratio[:, None] * matH_intra_acc
    
    vecN_comb_denom_val = tl.maximum(
        tl.abs(vecN_inter_acc + vecM_comb_ratio * vecN_intra_acc),
        tl.exp(MINIMUM_MAX_VAL - vecM_combine_val),
    )
    
    # Efficient division using vector operations
    matH_comb_val = matH_comb_num_val / (vecN_comb_denom_val[:, None] + EPS)
    
    # store the outputs - optimized for contiguous stores
    matHout_offset_x = idx_b_NC * L + idx_b_LQ * siz_b_LQ
    matHout_offset_y = idx_b_DHHV * siz_b_DHHV
    
    matHout_ptr = matHout + idx_b_BNH * str_matHV_B_NH
    matHout_ptr += matHout_offset_x * str_matHV_S + matHout_offset_y * str_matHV_DHHV
    
    # Use sequential store operations for better memory access
    tl.store(
        matHout_ptr + tl.arange(0, siz_b_LQ)[:, None] * str_matHV_S +
        tl.arange(0, siz_b_DHHV)[None, :] * str_matHV_DHHV,
        matH_comb_val.to(OUTPUT_DTYPE)
    )
    
    # Store vector outputs
    vecNout_ptr = (
        vecNout + idx_b_BNH * str_vecMN_B_NH +
        (idx_b_NC * L + idx_b_LQ * siz_b_LQ + tl.arange(0, siz_b_LQ)) * str_vecMN_S
    )
    
    vecMout_ptr = (
        vecMout + idx_b_BNH * str_vecMN_B_NH +
        (idx_b_NC * L + idx_b_LQ * siz_b_LQ + tl.arange(0, siz_b_LQ)) * str_vecMN_S
    )
    
    tl.store(vecNout_ptr, vecN_comb_denom_val.to(tl.float32))
    tl.store(vecMout_ptr, vecM_combine_val.to(tl.float32)) 