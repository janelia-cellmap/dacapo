from dacapo.experiments.datasplits import (
    TrainValidateDataSplitConfig,
    DummyDataSplitConfig,
)
from dacapo.experiments.datasplits.datasets import RawGTDatasetConfig
from dacapo.experiments.datasplits.datasets.arrays import (
    ZarrArrayConfig,
    BinarizeArrayConfig,
)

import zarr
import numpy as np

import pytest


@pytest.fixture()
def dummy_datasplit():
    yield DummyDataSplitConfig(name="dummy_datasplit")


@pytest.fixture()
def twelve_class_datasplit(tmp_path):
    """
    two crops for training, one for validation. Raw data is normally distributed
    around 0 with std 1.
    gt has 12 classes where class i in [0, 11] is all voxels with raw intensity
    between (raw.min() + i(raw.max()-raw.min())/12, raw.min() + (i+1)(raw.max()-raw.min())/12)
    """
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
            raw.dataset, shape=(40, 20, 20), dtype=np.float32
        )
        gt_dataset = twelve_class_zarr.create_dataset(
            gt.dataset, shape=(40, 20, 20), dtype=np.uint8
        )
        random_data = np.random.rand(40, 20, 20)
        # as intensities increase so does the class
        for i in list(np.linspace(random_data.min(), random_data.max(), 13))[1:]:
            gt_dataset[:] += random_data > i
        raw_dataset[:] = random_data
        raw_dataset.attrs["offset"] = (0, 0, 0)
        raw_dataset.attrs["resolution"] = (2, 2, 2)
        raw_dataset.attrs["axis_names"] = ("z", "y", "x")
        gt_dataset.attrs["offset"] = (0, 0, 0)
        gt_dataset.attrs["resolution"] = (2, 2, 2)
        gt_dataset.attrs["axis_names"] = ("z", "y", "x")

    crop1 = RawGTDatasetConfig(name="crop1", raw_config=crop1_raw, gt_config=crop1_gt)
    crop2 = RawGTDatasetConfig(name="crop2", raw_config=crop2_raw, gt_config=crop2_gt)
    crop3 = RawGTDatasetConfig(name="crop3", raw_config=crop3_raw, gt_config=crop3_gt)

    twelve_class_datasplit_config = TrainValidateDataSplitConfig(
        name="twelve_class_datasplit",
        train_configs=[crop1],
        validate_configs=[crop2, crop3],
    )
    yield twelve_class_datasplit_config


@pytest.fixture()
def six_class_datasplit(tmp_path):
    """
    two crops for training, one for validation. Raw data is normally distributed
    around 0 with std 1.
    gt is provided as distances. First, gt is generated as a 12 class problem:
    gt has 12 classes where class i in [0, 11] is all voxels with raw intensity
    between (raw.min() + i(raw.max()-raw.min())/12, raw.min() + (i+1)(raw.max()-raw.min())/12).
    Then we pair up classes (i, i+1) for i in [0,2,4,6,8,10], and compute distances to
    the nearest voxel in the pair. This leaves us with 6 distance channels.
    """
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
    crop1_distances = BinarizeArrayConfig(
        "crop1_distances",
        source_array_config=crop1_gt,
        groupings=[
            ("a", [0, 1]),
            ("b", [2, 3]),
            ("c", [4, 5]),
            ("d", [6, 7]),
            ("e", [8, 9]),
            ("f", [10, 11]),
        ],
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
    crop2_distances = BinarizeArrayConfig(
        "crop2_distances",
        source_array_config=crop2_gt,
        groupings=[
            ("a", [0, 1]),
            ("b", [2, 3]),
            ("c", [4, 5]),
            ("d", [6, 7]),
            ("e", [8, 9]),
            ("f", [10, 11]),
        ],
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
    crop3_distances = BinarizeArrayConfig(
        "crop3_distances",
        source_array_config=crop3_gt,
        groupings=[
            ("a", [0, 1]),
            ("b", [2, 3]),
            ("c", [4, 5]),
            ("d", [6, 7]),
            ("e", [8, 9]),
            ("f", [10, 11]),
        ],
    )
    for raw, gt in zip(
        [crop1_raw, crop2_raw, crop3_raw], [crop1_gt, crop2_gt, crop3_gt]
    ):
        raw_dataset = twelve_class_zarr.create_dataset(
            raw.dataset, shape=(40, 20, 20), dtype=np.float32
        )
        gt_dataset = twelve_class_zarr.create_dataset(
            gt.dataset, shape=(40, 20, 20), dtype=np.uint8
        )
        random_data = np.random.rand(40, 20, 20)
        # as intensities increase so does the class
        for i in list(np.linspace(random_data.min(), random_data.max(), 13))[1:]:
            gt_dataset[:] += random_data > i
        raw_dataset[:] = random_data
        raw_dataset.attrs["offset"] = (0, 0, 0)
        raw_dataset.attrs["resolution"] = (2, 2, 2)
        raw_dataset.attrs["axis_names"] = ("z", "y", "x")
        gt_dataset.attrs["offset"] = (0, 0, 0)
        gt_dataset.attrs["resolution"] = (2, 2, 2)
        gt_dataset.attrs["axis_names"] = ("z", "y", "x")

    crop1 = RawGTDatasetConfig(
        name="crop1", raw_config=crop1_raw, gt_config=crop1_distances
    )
    crop2 = RawGTDatasetConfig(
        name="crop2", raw_config=crop2_raw, gt_config=crop2_distances
    )
    crop3 = RawGTDatasetConfig(
        name="crop3", raw_config=crop3_raw, gt_config=crop3_distances
    )

    six_class_distances_datasplit_config = TrainValidateDataSplitConfig(
        name="six_class_distances_datasplit",
        train_configs=[crop1, crop2],
        validate_configs=[crop3],
    )
    return six_class_distances_datasplit_config
