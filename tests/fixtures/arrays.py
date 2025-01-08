from dacapo.experiments.datasplits.datasets.arrays import (
    ZarrArrayConfig,
    BinarizeArrayConfig,
    DummyArrayConfig,
)

import zarr
import numpy as np
from numcodecs import Zstd
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
    
@pytest.fixture(scope="session")
def metadata_ome_zarr():
    zarr_metadata = {
        "multiscales": [
            {
                "axes": [
                    {"name": "z", "type": "space", "unit": "nanometer"},
                    {"name": "y", "type": "space", "unit": "nanometer"},
                    {"name": "x", "type": "space", "unit": "nanometer"},
                ],
                "coordinateTransformations": [],
                "datasets": [
                        {
                        "coordinateTransformations": [
                            {"scale": [4.2,7.4,5.6], "type": "scale"},
                            {"translation": [6.0,10.0,2.0],"type": "translation"}
                        ],
                        "path": "s0"
                        },
                        {
                        "coordinateTransformations": [
                            {"type": "scale","scale": [8.4,14.8,11.2]},
                            {"type": "translation","translation": [8.1,13.7,4.8]}
                        ],
                        "path": "s1"
                        }
                ],
                "name": "multiscale_dataset",
                "version": "0.4",
            }
        ]
    }
    return zarr_metadata

    
@pytest.fixture(scope="session")
def multiscale_dataset(tmp_path, metadata_ome_zarr):
    path = tmp_path.mktemp("test", numbered=False)
    test_zarr_path = path / "multiscale_zarr_dataset.zarr"
    
    store = zarr.DirectoryStore(test_zarr_path)
    multiscale_group = zarr.group(store=store, path="multiscale_dataset", overwrite=True)

    for level in [0,1]:
        scaling = pow(2, level)
        multiscale_group.require_dataset(
            name=f's{level}',
            shape=(100/scaling, 80/scaling, 60/scaling),
            chunks=10,
            dtype=np.float32,
            compressor=Zstd(level=6),
        )

    multiscale_group.attrs.update(metadata_ome_zarr)

    yield multiscale_group
