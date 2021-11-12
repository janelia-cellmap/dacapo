from dacapo.experiments.datasplits.datasets.arrays import (
    ZarrArrayConfig,
    CellMapArrayConfig,
)

import zarr
import numpy as np


def mk_cellmap_array(temp_path):
    zarr_array_config = ZarrArrayConfig(
        name="zarr_array",
        file_name=temp_path / "zarr_array.zarr",
        dataset="volumes/test",
    )
    zarr_container = zarr.open(str(temp_path / "zarr_array.zarr"))
    dataset = zarr_container.create_dataset(
        "volumes/test", data=np.arange(0, 100).reshape(10, 5, 2)
    )
    dataset.attrs["offset"] = (12, 12, 12)
    dataset.attrs["resolution"] = (1, 2, 4)
    dataset.attrs["axes"] = "zyx"

    cellmap_array_config = CellMapArrayConfig(
        name="cellmap_zarr_array",
        source_array_config=zarr_array_config,
        groupings=[list(range(0, 10)), list(range(10, 70)), list(range(70, 90))],
    )

    return cellmap_array_config
