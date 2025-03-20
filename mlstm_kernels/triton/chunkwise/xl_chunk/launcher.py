# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import torch
import triton
import logging
import triton.language as tl

try:
    from ...amd_optimizations import is_amd, get_amd_kernel_config, get_amd_warp_size, tune_amd_memory_access, optimize_grid_for_amd
    AMD_SUPPORT = True
except ImportError:
    AMD_SUPPORT = False
    
    def is_amd():
        return False
    
    def get_amd_kernel_config(head_dim_q, head_dim_v=None):
        return {}
    
    def get_amd_warp_size():
        return 32
    
    def tune_amd_memory_access(head_dim):
        return {"vectorize_load": False, "vectorize_width": 1, "use_shared_memory": True, "prefetch_factor": 1}
    
    def optimize_grid_for_amd(grid_fn):
        return grid_fn

from ..fw_kernel import mlstm_chunkwise__parallel_fw_Hintra_kernel

# dtype mapping
dtype_to_triton = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
}

class MLSTMChunkwiseParallelFwdHintraOp:
    def __call__(
        self,
        matQ: torch.Tensor,
        matK: torch.Tensor,
        matV: torch.Tensor,
        matC_states: torch.Tensor,
        vecN_states: torch.Tensor,
        scaMinter_states: torch.Tensor,
        vecI: torch.Tensor,
        vecB: torch.Tensor,
        chunk_size: int,
        qk_scale: float,
        eps: float = 0.0,
    ):
        B = matQ.shape[0]  # batch size
        NH = matQ.shape[1]  # num_heads
        S = matQ.shape[2]  # sequence length
        DHQK = matQ.shape[3]  # head dimension q/k
        DHHV = matV.shape[3]  # head dimension v
        
        assert S % chunk_size == 0, f"sequence length {S} must be divisible by chunk_size {chunk_size}"
        L = chunk_size
        NC = S // L  # number of chunks
        
        # Check for AMD hardware and use optimized kernel if available
        do_amd_optimizations = AMD_SUPPORT and is_amd()
        
        # Create output tensors
        matHout = torch.empty_like(matV)
        vecNout = torch.empty(B, NH, S, dtype=torch.float32, device=matQ.device)
        vecMout = torch.empty(B, NH, S, dtype=torch.float32, device=matQ.device)
        
        # Special handling for head_dim=64 which needs careful optimization
        is_head_dim_64 = DHQK == 64
        
        if do_amd_optimizations:
            try:
                # Import AMD-optimized kernel
                from ...chunkwise.xl_chunk.fw_kernel_amd import mlstm_chunkwise__parallel_fw_Hintra_kernel_amd
                
                # For head_dim=64, use our specialized kernel based on empirical results
                if is_head_dim_64:
                    try:
                        from ...chunkwise.xl_chunk.fw_kernel_amd_dim64 import mlstm_chunkwise__parallel_fw_Hintra_kernel_amd_dim64
                        
                        # Based on benchmark results, use specialized kernel only for these configurations:
                        # 1. Batch size 1, seq length 2048
                        # 2. Batch size 4, seq length 2048
                        use_specialized_kernel = False
                        
                        if B == 1 and S == 2048:
                            use_specialized_kernel = True
                            logging.info(f"Using specialized AMD kernel for head_dim=64, batch={B}, seq_len={S}")
                        elif B == 4 and S == 2048:
                            use_specialized_kernel = True
                            logging.info(f"Using specialized AMD kernel for head_dim=64, batch={B}, seq_len={S}")
                        else:
                            logging.info(f"Using stock kernel for head_dim=64 configuration batch={B}, seq_len={S}")
                        
                        if use_specialized_kernel:
                            # Use smaller block sizes specifically for head_dim=64
                            siz_b_LQ = min(L, 32)  # Smaller blocks for head_dim=64
                            siz_b_LKV = min(L, 32)
                            siz_b_DHHV = 32
                            siz_b_DHQK = 32
                            
                            # Define optimized grid with more blocks for better parallelism
                            grid = (
                                triton.cdiv(DHHV, siz_b_DHHV) * 2,  # Double grid size
                                triton.cdiv(L, siz_b_LQ) * 2,       # for better parallelism
                                NC * B * NH,
                            )
                            
                            # Call the specialized kernel for head_dim=64
                            mlstm_chunkwise__parallel_fw_Hintra_kernel_amd_dim64[grid](
                                matQ,
                                matK,
                                matV,
                                matC_states,
                                vecN_states,
                                scaMinter_states,
                                vecI,
                                vecB,
                                matHout,
                                vecNout,
                                vecMout,
                                qk_scale,
                                matQ.stride(0) * matQ.stride(1),
                                matQ.stride(2),
                                matQ.stride(3),
                                matV.stride(0) * matV.stride(1),
                                matV.stride(2),
                                matV.stride(3),
                                matC_states.stride(0) * matC_states.stride(1),
                                matC_states.stride(2),
                                matC_states.stride(3),
                                vecN_states.stride(0) * vecN_states.stride(1),
                                vecN_states.stride(2),
                                scaMinter_states.stride(0) * scaMinter_states.stride(1),
                                vecI.stride(0) * vecI.stride(1),
                                vecI.stride(2),
                                vecI.stride(3),
                                vecNout.stride(0) * vecNout.stride(1),
                                vecNout.stride(2),
                                B,
                                NH,
                                S,
                                DHQK,
                                DHHV,
                                NC,
                                L,
                                siz_b_LQ,
                                siz_b_LKV,
                                siz_b_DHQK,
                                siz_b_DHHV,
                                DTYPE=dtype_to_triton[matQ.dtype],
                                OUTPUT_DTYPE=dtype_to_triton[matQ.dtype],
                                EPS=eps,
                            )
                            
                            return matHout, vecNout, vecMout
                        
                        # For other configurations, fall through to standard AMD kernel
                        
                    except (ImportError, AttributeError) as e:
                        logging.warning(f"Specialized kernel for head_dim=64 not available: {e}. Using standard AMD kernel.")
                
                # Get optimized configuration for AMD GPUs
                amd_kernel_config = get_amd_kernel_config(DHQK, DHHV)
                warp_size = get_amd_warp_size()
                memory_config = tune_amd_memory_access(DHQK)
                
                # Set block sizes based on AMD optimizations
                siz_b_DHHV = min(DHHV, amd_kernel_config.get("BLOCK_SIZE_V", 64))
                siz_b_DHQK = min(DHQK, amd_kernel_config.get("BLOCK_SIZE_Q", 64))
                
                # For head_dim=64, we use smaller block sizes
                if is_head_dim_64:
                    siz_b_LQ = min(L, 32)  # Smaller blocks for head_dim=64
                    siz_b_LKV = min(L, 32)
                    
                    # Ensure these are not rounded up to multiples of 16
                    # as smaller values work better for head_dim=64
                    siz_b_DHHV = min(siz_b_DHHV, 32)
                    siz_b_DHQK = min(siz_b_DHQK, 32)
                else:
                    siz_b_LQ = min(L, warp_size)
                    siz_b_LKV = min(L, warp_size)
                
                # Ensure block sizes are multiples of 16 for better memory alignment
                # but for head_dim=64 we prefer smaller blocks
                if not is_head_dim_64:
                    siz_b_DHHV = max(16, ((siz_b_DHHV + 15) // 16) * 16)
                    siz_b_DHQK = max(16, ((siz_b_DHQK + 15) // 16) * 16)
                    siz_b_LQ = max(16, ((siz_b_LQ + 15) // 16) * 16)
                    siz_b_LKV = max(16, ((siz_b_LKV + 15) // 16) * 16)
                
                # Define optimized grid calculation function
                def get_grid():
                    if is_head_dim_64:
                        # For head_dim=64, use more, smaller grid elements
                        return (
                            triton.cdiv(DHHV, siz_b_DHHV) * 2,  # Increase grid size
                            triton.cdiv(L, siz_b_LQ) * 2,       # for better parallelism
                            NC * B * NH,
                        )
                    else:
                        return (
                            triton.cdiv(DHHV, siz_b_DHHV),
                            triton.cdiv(L, siz_b_LQ),
                            NC * B * NH,
                        )
                
                # Apply AMD-specific grid optimization
                optimized_grid = optimize_grid_for_amd(get_grid)(head_dim=DHQK)
                
                logging.info(f"Using AMD-optimized kernel with block sizes: "
                             f"DHHV={siz_b_DHHV}, DHQK={siz_b_DHQK}, LQ={siz_b_LQ}, LKV={siz_b_LKV}")
                
                # Call AMD-optimized kernel
                mlstm_chunkwise__parallel_fw_Hintra_kernel_amd[optimized_grid](
                    matQ,
                    matK,
                    matV,
                    matC_states,
                    vecN_states,
                    scaMinter_states,
                    vecI,
                    vecB,
                    matHout,
                    vecNout,
                    vecMout,
                    qk_scale,
                    matQ.stride(0) * matQ.stride(1),
                    matQ.stride(2),
                    matQ.stride(3),
                    matV.stride(0) * matV.stride(1),
                    matV.stride(2),
                    matV.stride(3),
                    matC_states.stride(0) * matC_states.stride(1),
                    matC_states.stride(2),
                    matC_states.stride(3),
                    vecN_states.stride(0) * vecN_states.stride(1),
                    vecN_states.stride(2),
                    scaMinter_states.stride(0) * scaMinter_states.stride(1),
                    vecI.stride(0) * vecI.stride(1),
                    vecI.stride(2),
                    vecI.stride(3),
                    vecNout.stride(0) * vecNout.stride(1),
                    vecNout.stride(2),
                    B,
                    NH,
                    S,
                    DHQK,
                    DHHV,
                    NC,
                    L,
                    siz_b_LQ,
                    siz_b_LKV,
                    siz_b_DHQK,
                    siz_b_DHHV,
                    DTYPE=dtype_to_triton[matQ.dtype],
                    OUTPUT_DTYPE=dtype_to_triton[matQ.dtype],
                    EPS=eps,
                    AMD_OPTIM=True,
                )
                
                return matHout, vecNout, vecMout
                
            except (ImportError, AttributeError) as e:
                logging.warning(f"AMD-optimized kernel not available or failed to launch: {e}. Falling back to standard kernel.")
        
        # For head_dim=64 on AMD hardware, the stock kernel might be faster
        # So we fall back to the standard CUDA kernel when our optimizations don't show benefit
        if is_head_dim_64 and not do_amd_optimizations:
            logging.info("Using stock kernel for head_dim=64 configuration")
        
        # Standard CUDA kernel (fallback)
        # Calculate block sizes
        siz_b_DHHV = min(DHHV, 64)
        siz_b_DHQK = min(DHQK, 64)
        siz_b_LQ = min(L, 64)
        siz_b_LKV = min(L, 64)
        
        # Simple grid
        grid = (
            triton.cdiv(DHHV, siz_b_DHHV),
            triton.cdiv(L, siz_b_LQ),
            NC * B * NH,
        )
        
        # Call standard kernel
        mlstm_chunkwise__parallel_fw_Hintra_kernel[grid](
            matQ,
            matK,
            matV,
            matC_states,
            vecN_states,
            scaMinter_states,
            vecI,
            vecB,
            matHout,
            vecNout,
            vecMout,
            qk_scale,
            matQ.stride(0) * matQ.stride(1),
            matQ.stride(2),
            matQ.stride(3),
            matV.stride(0) * matV.stride(1),
            matV.stride(2),
            matV.stride(3),
            matC_states.stride(0) * matC_states.stride(1),
            matC_states.stride(2),
            matC_states.stride(3),
            vecN_states.stride(0) * vecN_states.stride(1),
            vecN_states.stride(2),
            scaMinter_states.stride(0) * scaMinter_states.stride(1),
            vecI.stride(0) * vecI.stride(1),
            vecI.stride(2),
            vecI.stride(3),
            vecNout.stride(0) * vecNout.stride(1),
            vecNout.stride(2),
            B,
            NH,
            S,
            DHQK,
            DHHV,
            NC,
            L,
            siz_b_LQ,
            siz_b_LKV,
            siz_b_DHQK,
            siz_b_DHHV,
            DTYPE=dtype_to_triton[matQ.dtype],
            OUTPUT_DTYPE=dtype_to_triton[matQ.dtype],
            EPS=eps,
        )
        
        return matHout, vecNout, vecMout 