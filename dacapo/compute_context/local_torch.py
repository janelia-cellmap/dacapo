from .compute_context import ComputeContext

import torch
import attr

from typing import Optional


@attr.s
class LocalTorch(ComputeContext):
    _device: Optional[str] = attr.ib(
        default=None,
        metadata={
            "help_text": "The device to run on. One of: 'cuda', 'cpu'. Can also be left undefined. "
            "If undefined we will use 'cuda' if possible or fall back on 'cpu'."
        },
    )

    @property
    def device(self):
        if self._device is None:
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        return torch.device(self._device)
