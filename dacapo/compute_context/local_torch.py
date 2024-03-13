from .compute_context import ComputeContext
import torch
import attr

from typing import Optional


@attr.s
class LocalTorch(ComputeContext):
    """
    The LocalTorch class is a subclass of the ComputeContext class.
    It is used to specify the context in which computations are to be done.
    LocalTorch is used to specify that computations are to be done on the local machine using PyTorch.

    Attributes:
        _device (Optional[str]): This stores the type of device on which torch computations are to be done. It can
        take "cuda" for GPU or "cpu" for CPU. None value results in automatic detection of device type.
    """

    _device: Optional[str] = attr.ib(
        default=None,
        metadata={
            "help_text": "The device to run on. One of: 'cuda', 'cpu'. Can also be left undefined. "
            "If undefined we will use 'cuda' if possible or fall back on 'cpu'."
        },
    )

    @property
    def device(self):
        """
        A property method that returns the torch device object. It automatically detects and uses "cuda" (GPU) if
        available, else it falls back on using "cpu".
        """
        if self._device is None:
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        return torch.device(self._device)
