from dacapo.experiments.datasplits.datasets.arrays import (
    ZarrArrayConfig,
    BinarizeArrayConfig,
    DummyArrayConfig,
)

import zarr
import numpy as np

import pytest


@pytest.fixture()
def dummy_array():
    yield DummyArrayConfig(name="dummy_array")


@pytest.fixture()
def zarr_array(tmp_path):
    zarr_array_config = ZarrArrayConfig(
        name="zarr_array",
        file_name=tmp_path / "zarr_array.zarr",
        dataset="volumes/test",
    )
    zarr_container = zarr.open(str(zarr_array_config.file_name))
    dataset = zarr_container.create_dataset(
        zarr_array_config.dataset, data=np.zeros((100, 50, 25), dtype=np.float32)
    )
    dataset.attrs["offset"] = (12, 12, 12)
    dataset.attrs["resolution"] = (1, 2, 4)
    dataset.attrs["axis_names"] = ["z", "y", "x"]
    yield zarr_array_config


@pytest.fixture()
def cellmap_array(tmp_path):
    zarr_array_config = ZarrArrayConfig(
        name="zarr_array",
        file_name=tmp_path / "zarr_array.zarr",
        dataset="volumes/test",
    )
    zarr_container = zarr.open(str(zarr_array_config.file_name))
    dataset = zarr_container.create_dataset(
        zarr_array_config.dataset,
        data=np.arange(0, 100, dtype=np.uint8).reshape(10, 5, 2),
    )
    dataset.attrs["offset"] = (12, 12, 12)
    dataset.attrs["resolution"] = (1, 2, 4)
    dataset.attrs["axis_names"] = ["z", "y", "x"]

    cellmap_array_config = BinarizeArrayConfig(
        name="cellmap_zarr_array",
        source_array_config=zarr_array_config,
        groupings=[
            ("a", list(range(0, 10))),
            ("b", list(range(10, 70))),
            ("c", list(range(70, 90))),
        ],
    )

    yield cellmap_array_config
