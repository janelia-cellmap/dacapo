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
        oom_limit (Optional[float | int]): The out of GPU memory to leave free in GB. If the free memory is below
        this limit, we will fall back on CPU.
    Methods:
        device(): Returns the torch device object.
    Note:
        The class is a subclass of the ComputeContext class.
    """

    distribute_workers: Optional[bool] = attr.ib(
        default=False,
        metadata={
            "help_text": "Whether to distribute the workers across multiple nodes or processes."
        },
    )
    _device: Optional[str] = attr.ib(
        default=None,
        metadata={
            "help_text": "The device to run on. One of: 'cuda', 'cpu'. Can also be left undefined. "
            "If undefined we will use 'cuda' if possible or fall back on 'cpu'."
        },
    )

    oom_limit: Optional[float | int] = attr.ib(
        default=4.2,
        metadata={
            "help_text": "The out of GPU memory to leave free in GB. If the free memory is below this limit, we will fall back on CPU."
        },
    )

    @property
    def device(self):
        """
        A property method that returns the torch device object. It automatically detects and uses "cuda" (GPU) if
        available, else it falls back on using "cpu".

        Returns:
            torch.device: The torch device object.
        """
        if self._device is None:
            if torch.cuda.is_available():
                # TODO: make this more sophisticated, for multiple GPUs for instance
                free = torch.cuda.mem_get_info()[0] / 1024**3
                if free < self.oom_limit:  # less than 1 GB free, decrease chance of OOM
                    return torch.device("cpu")
                return torch.device("cuda")
            # Multiple MPS ops are not available yet : https://github.com/pytorch/pytorch/issues/77764
            # got error aten::max_pool3d_with_indices
            # can be back when mps is fixed
            # elif torch.backends.mps.is_available():
            #     return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self._device)
