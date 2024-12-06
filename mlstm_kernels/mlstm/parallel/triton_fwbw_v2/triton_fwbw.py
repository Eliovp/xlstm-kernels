# Copyright JKU Linz 2024
# Author: Maximilian Beck
from collections.abc import Callable

import torch
from torch.amp import custom_bwd, custom_fwd

from ....torch.utils import contiguous
from ._triton_bw import mlstm_bw
from ._triton_fw import mlstm_fw


def _mlstm_parallel_fwbw_generator(autocast_kernel_dtype=torch.float16) -> Callable:
    class _mlstm_parallel_fwbw(torch.autograd.Function):
        @staticmethod
        @custom_fwd(device_type="cuda", cast_inputs=autocast_kernel_dtype)
        @contiguous
        def forward(
            ctx,
            matQ: torch.Tensor,
            matK: torch.Tensor,
            matV: torch.Tensor,
            vecI: torch.Tensor,
            vecF: torch.Tensor,
            eps: float = 1e-6,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            matH, vecM, vecN = mlstm_fw(
                matQ=matQ,
                matK=matK,
                matV=matV,
                vecI=vecI,
                vecF=vecF,
                eps=eps,
            )
            ctx.save_for_backward(matQ, matK, matV, vecI, vecF, vecM, vecN)
            return matH, vecM, vecN

        @staticmethod
        @custom_bwd(device_type="cuda")
        @contiguous
        def backward(
            ctx,
            matDeltaHtilde: torch.Tensor,
            vecDeltaM_unused: torch.Tensor,
            vecDeltaN_unused: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            (matQ, matK, matV, vecI, vecF, vecM, vecN) = ctx.saved_tensors
            matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaF = mlstm_bw(
                matDeltaHtilde=matDeltaHtilde,
                matQ=matQ,
                matK=matK,
                matV=matV,
                vecI=vecI,
                vecF=vecF,
                vecM=vecM,
                vecN=vecN,
            )
            return matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaF, None

    return _mlstm_parallel_fwbw


_mlstm_parallel_fwbw_float32 = _mlstm_parallel_fwbw_generator(autocast_kernel_dtype=torch.float32)
_mlstm_parallel_fwbw_float16 = _mlstm_parallel_fwbw_generator(autocast_kernel_dtype=torch.float16)
_mlstm_parallel_fwbw_bfloat16 = _mlstm_parallel_fwbw_generator(autocast_kernel_dtype=torch.bfloat16)


def _get_parallel_fwbw_kernel(autocast_kernel_dtype: torch.dtype) -> Callable:
    if autocast_kernel_dtype == torch.float32:
        return _mlstm_parallel_fwbw_float32
    elif autocast_kernel_dtype == torch.float16:
        return _mlstm_parallel_fwbw_float16
    elif autocast_kernel_dtype == torch.bfloat16:
        return _mlstm_parallel_fwbw_bfloat16
    else:
        raise ValueError(f"Unsupported autocast_kernel_dtype: {autocast_kernel_dtype}")


def mlstm_parallel_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    c_initial: torch.Tensor = None,
    n_initial: torch.Tensor = None,
    m_initial: torch.Tensor = None,
    return_last_states: bool = False,
    eps: float = 1e-6,
    autocast_kernel_dtype: torch.dtype = torch.float32,
    **kwargs,
) -> torch.Tensor:
    assert c_initial is None, "c_initial is not supported"
    assert n_initial is None, "n_initial is not supported"
    assert m_initial is None, "m_initial is not supported"
    assert return_last_states is False, "return_last_states is not supported"

    _mlstm_parallel_fwbw = _get_parallel_fwbw_kernel(autocast_kernel_dtype=autocast_kernel_dtype)

    matH, _, _ = _mlstm_parallel_fwbw.apply(q, k, v, i, f, eps)
    return matH
