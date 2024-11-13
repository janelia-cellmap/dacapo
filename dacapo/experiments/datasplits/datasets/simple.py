from .dataset_config import DatasetConfig

from funlib.persistence import Array, open_ds


import attr

from pathlib import Path
import numpy as np


@attr.s
class SimpleDataset(DatasetConfig):
    path: Path = attr.ib()
    weight: int = attr.ib(default=1)
    raw_name: str = attr.ib(default="raw")
    gt_name: str = attr.ib(default="labels")
    mask_name: str = attr.ib(default="mask")

    @staticmethod
    def dataset_type(dataset_config):
        return dataset_config

    @property
    def raw(self) -> Array:
        raw_array = open_ds(self.path / self.raw_name)
        dtype = raw_array.dtype
        if dtype == np.uint8:
            raw_array.lazy_op(lambda data: data.astype(np.float32) / 255)
        elif dtype == np.uint16:
            raw_array.lazy_op(lambda data: data.astype(np.float32) / 65535)
        elif np.issubdtype(dtype, np.floating):
            pass
        elif np.issubdtype(dtype, np.integer):
            raise Exception(
                f"Not sure how to normalize intensity data with dtype {dtype}"
            )
        return raw_array

    @property
    def gt(self) -> Array:
        return open_ds(self.path / self.gt_name)

    @property
    def mask(self) -> Array | None:
        mask_path = self.path / self.mask_name
        if mask_path.exists():
            mask = open_ds(mask_path)
            assert np.issubdtype(mask.dtype, np.integer), "Mask must be integer type"
            mask.lazy_op(lambda data: data > 0)
            return mask
        return None

    @property
    def sample_points(self) -> None:
        return None

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name
