from .torch_fw import mlstm_chunkwise_parallel_fw_parallel as mlstm_torch_autograd
from .triton_fwbw import mlstm_fwbw

registry = {
    "torch_autograd": mlstm_torch_autograd,
    "triton": mlstm_fwbw,
}
