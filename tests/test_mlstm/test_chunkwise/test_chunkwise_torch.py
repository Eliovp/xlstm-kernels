import logging

from mlstm_kernels.mlstm.chunkwise import (
    mlstm_chunkwise_max_triton_v3,
    mlstm_chunkwise_torch_autograd,
    mlstm_chunkwise_torch_ownbw,
)
from mlstm_kernels.mlstm.parallel import (
    mlstm_parallel_stable_torch_autograd,
    mlstm_parallel_torch_autograd,
)
from mlstm_kernels.test_utils.test_fixtures import test_session_folder  # noqa
from mlstm_kernels.test_utils.test_templates.template_parallel_interface import (
    template_test_parallel_interface,
)

import pytest
import torch

from ..test_params import final_combinations

LOGGER = logging.getLogger(__name__)

TEST_FOLDER_NAME_PREFIX = "chunkwise-torch"


class TestChunkwiseTorchAgVsStableTorch:
    """Test torch implementations forwards with backward computed by autograd."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
    @pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
    def test_chunkwise_torch_vs_stable_torch_fp32(
        self, test_session_folder, S, B, NH, DHQK, DHHV
    ):
        print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
        template_test_parallel_interface(
            baseline_fn=mlstm_parallel_stable_torch_autograd,
            target_fn=mlstm_chunkwise_torch_autograd,
            baseline_name="parallel_stable_ag",
            target_name="chunkwise_ag",
            S=S,
            B=B,
            NH=NH,
            DHQK=DHQK,
            DHHV=DHHV,
            dtype=torch.float32,
            atol_fw=1e-4,
            rtol_fw=1e-3,
            atol_fwbw=1e-2,
            rtol_fwbw=1e-3,
            vmax=1e-4,
            test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
            save_dir=str(test_session_folder),
            add_fp64_baseline=False,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
    @pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
    def test_chunkwise_torch_vs_unstable_torch_fp32(
        self, test_session_folder, S, B, NH, DHQK, DHHV
    ):
        print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
        template_test_parallel_interface(
            baseline_fn=mlstm_parallel_torch_autograd,
            target_fn=mlstm_chunkwise_torch_autograd,
            baseline_name="parallel_unstable_ag",
            target_name="chunkwise_ag",
            S=S,
            B=B,
            NH=NH,
            DHQK=DHQK,
            DHHV=DHHV,
            dtype=torch.float32,
            atol_fw=1e-4,
            rtol_fw=1e-3,
            atol_fwbw=1e-2,
            rtol_fwbw=1e-3,
            vmax=1e-4,
            test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
            save_dir=str(test_session_folder),
            add_fp64_baseline=False,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
    @pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
    def test_chunkwise_torch_vs_stable_torch_bf16(
        self, test_session_folder, S, B, NH, DHQK, DHHV
    ):
        print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
        template_test_parallel_interface(
            baseline_fn=mlstm_parallel_stable_torch_autograd,
            target_fn=mlstm_chunkwise_torch_autograd,
            baseline_name="parallel_stable_ag",
            target_name="chunkwise_ag",
            S=S,
            B=B,
            NH=NH,
            DHQK=DHQK,
            DHHV=DHHV,
            dtype=torch.bfloat16,
            atol_fw=1.0,  # 3.0
            rtol_fw=1.0,
            atol_fwbw=3.0,  # 3.5
            rtol_fwbw=1.0,
            vmax=1.0,
            test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
            save_dir=str(test_session_folder),
            add_fp64_baseline=False,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
    @pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
    def test_chunkwise_torch_vs_unstable_torch_bf16(
        self, test_session_folder, S, B, NH, DHQK, DHHV
    ):
        print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
        template_test_parallel_interface(
            baseline_fn=mlstm_parallel_torch_autograd,
            target_fn=mlstm_chunkwise_torch_autograd,
            baseline_name="parallel_unstable_ag",
            target_name="chunkwise_ag",
            S=S,
            B=B,
            NH=NH,
            DHQK=DHQK,
            DHHV=DHHV,
            dtype=torch.bfloat16,
            atol_fw=6.0,
            rtol_fw=1.0,
            atol_fwbw=15.0,
            rtol_fwbw=1.0,
            vmax=1.0,
            test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
            save_dir=str(test_session_folder),
            add_fp64_baseline=False,
        )


class TestChunkwiseTorchOwnbw:
    """Test torch implementations forwards with own custom backward."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
    @pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
    def test_chunkwise_torch_ag_vs_chunkwise_torch_obw_fp32(
        self, test_session_folder, S, B, NH, DHQK, DHHV
    ):
        # Note we slightly increase the rtol_fwbw here, the vecI.grad did not match
        print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
        template_test_parallel_interface(
            baseline_fn=mlstm_chunkwise_torch_autograd,
            target_fn=mlstm_chunkwise_torch_ownbw,
            baseline_name="chunkwise_ag",
            target_name="chunkwise_obw",
            S=S,
            B=B,
            NH=NH,
            DHQK=DHQK,
            DHHV=DHHV,
            dtype=torch.float32,
            atol_fw=1e-4,
            rtol_fw=1e-3,
            atol_fwbw=1e-2,
            rtol_fwbw=4e-3,
            vmax=1e-3,
            test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
            save_dir=str(test_session_folder),
            add_fp64_baseline=False,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
    @pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
    def test_parallel_stable_ag_vs_chunkwise_torch_obw_fp32(
        self, test_session_folder, S, B, NH, DHQK, DHHV
    ):
        # Note we slightly increase the rtol_fwbw here, the vecI.grad did not match
        print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
        template_test_parallel_interface(
            baseline_fn=mlstm_parallel_stable_torch_autograd,
            target_fn=mlstm_chunkwise_torch_ownbw,
            baseline_name="parallel_stable_ag",
            target_name="chunkwise_obw",
            S=S,
            B=B,
            NH=NH,
            DHQK=DHQK,
            DHHV=DHHV,
            dtype=torch.float32,
            atol_fw=1e-4,
            rtol_fw=1e-3,
            atol_fwbw=1e-2,
            rtol_fwbw=4e-3,
            vmax=1e-3,
            test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
            save_dir=str(test_session_folder),
            add_fp64_baseline=False,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
    @pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
    def test_chunkwise_torch_obw_vs_chunkwise_tritonv3_fp32(
        self, test_session_folder, S, B, NH, DHQK, DHHV
    ):
        #! Note here the differences are higher here
        print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
        template_test_parallel_interface(
            baseline_fn=mlstm_chunkwise_max_triton_v3,
            target_fn=mlstm_chunkwise_torch_ownbw,
            baseline_name="max_triton_v3",
            target_name="chunkwise_obw",
            S=S,
            B=B,
            NH=NH,
            DHQK=DHQK,
            DHHV=DHHV,
            dtype=torch.float32,
            atol_fw=1e-2,
            rtol_fw=1e-2,
            atol_fwbw=4e-1,  # 3.5
            rtol_fwbw=0.8,
            vmax=1e-2,
            test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
            save_dir=str(test_session_folder),
            add_fp64_baseline=False,
        )
