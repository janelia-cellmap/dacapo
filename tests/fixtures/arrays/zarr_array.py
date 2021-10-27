from dacapo.experiments.datasplits.datasets.arrays import ZarrArrayConfig

import zarr
import numpy as np


def mk_zarr_array(temp_path):
    zarr_array_config = ZarrArrayConfig(
        name="zarr_array",
        file_name=temp_path / "zarr_array.zarr",
        dataset="volumes/test",
    )
    zarr_container = zarr.open(str(temp_path / "zarr_array.zarr"))
    dataset = zarr_container.create_dataset(
        "volumes/test", data=np.zeros((100, 50, 25))
    )
    dataset.attrs["offset"] = (12,12,12)
    dataset.attrs["resolution"] = (1, 2, 4)
    dataset.attrs["axes"] = "zyx"
    return zarr_array_config
