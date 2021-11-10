from dacapo.experiments.datasplits import TrainValidateDataSplitConfig
from dacapo.experiments.datasplits.datasets import RawGTDatasetConfig
from dacapo.experiments.datasplits.datasets.arrays import ZarrArrayConfig

import zarr
import numpy as np


def mk_twelve_class_datasplit(tmp_path):
    twelve_class_zarr = zarr.open(tmp_path / "twelve_class.zarr", "w")
    crop1_raw = ZarrArrayConfig(
        name="crop1_raw",
        file_name=tmp_path / "twelve_class.zarr",
        dataset=f"volumes/crop1/raw",
    )
    crop1_gt = ZarrArrayConfig(
        name="crop1_gt",
        file_name=tmp_path / "twelve_class.zarr",
        dataset=f"volumes/crop1/gt",
    )
    crop2_raw = ZarrArrayConfig(
        name="crop2_raw",
        file_name=tmp_path / "twelve_class.zarr",
        dataset=f"volumes/crop2/raw",
    )
    crop2_gt = ZarrArrayConfig(
        name="crop2_gt",
        file_name=tmp_path / "twelve_class.zarr",
        dataset=f"volumes/crop2/gt",
    )
    crop3_raw = ZarrArrayConfig(
        name="crop3_raw",
        file_name=tmp_path / "twelve_class.zarr",
        dataset=f"volumes/crop3/raw",
    )
    crop3_gt = ZarrArrayConfig(
        name="crop3_gt",
        file_name=tmp_path / "twelve_class.zarr",
        dataset=f"volumes/crop3/gt",
    )
    for raw, gt in zip(
        [crop1_raw, crop2_raw, crop3_raw], [crop1_gt, crop2_gt, crop3_gt]
    ):
        raw_dataset = twelve_class_zarr.create_dataset(
            raw.dataset, shape=(100, 100, 100), dtype=np.float32
        )
        gt_dataset = twelve_class_zarr.create_dataset(
            gt.dataset, shape=(100, 100, 100), dtype=np.uint8
        )
        random_data = np.random.randn(100, 100, 100)
        # as intensities increase so does the class
        for i in list(np.linspace(random_data.min(), random_data.max(), 13))[1:]:
            gt_dataset[:] += random_data > i
        raw_dataset[:] = random_data
        raw_dataset.attrs["offset"] = (0, 0, 0)
        raw_dataset.attrs["resolution"] = (2, 2, 2)
        gt_dataset.attrs["offset"] = (0, 0, 0)
        gt_dataset.attrs["resolution"] = (2, 2, 2)

    crop1 = RawGTDatasetConfig(name="crop1", raw_config=crop1_raw, gt_config=crop1_gt)
    crop2 = RawGTDatasetConfig(name="crop2", raw_config=crop2_raw, gt_config=crop2_gt)
    crop3 = RawGTDatasetConfig(name="crop3", raw_config=crop3_raw, gt_config=crop3_gt)

    twelve_class_datasplit_config = TrainValidateDataSplitConfig(
        name="twelve_class_datasplit",
        train_configs=[crop1, crop2],
        validate_configs=[crop3],
    )
    return twelve_class_datasplit_config
