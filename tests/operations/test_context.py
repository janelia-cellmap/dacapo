import torch
from dacapo.compute_context import create_compute_context
import pytest


@pytest.mark.parametrize("device", [""])
def test_create_compute_context(device):
    compute_context = create_compute_context()
    assert compute_context is not None
    assert compute_context.device is not None
    if torch.cuda.is_available():
        assert compute_context.device == torch.device(
            "cuda"
        ), "Model is not on CUDA when CUDA is available {}".format(
            compute_context.device
        )
    else:
        assert compute_context.device == torch.device(
            "cpu"
        ), "Model is not on CPU when CUDA is not available {}".format(
            compute_context.device
        )
