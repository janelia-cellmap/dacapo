from dacapo.experiments.tasks import TaskConfig
from pathlib import Path
from typing import List
from enum import Enum
import os
import numpy as np
from dacapo.experiments.datasplits.datasets.arrays import (
    ArrayConfig,
    ZarrArrayConfig,
    ZarrArray,
    ResampledArrayConfig,
    BinarizeArrayConfig,
    IntensitiesArrayConfig,
    IntensitiesArray,
    MissingAnnotationsMaskConfig,
    OnesArrayConfig,
    ConcatArrayConfig,
    LogicalOrArrayConfig,
    CropArrayConfig,
    MergeInstancesArrayConfig,
)
import logging

logger = logging.getLogger(__name__)

class DatasetType(Enum):
    val = 1
    train = 2


max_gt_downsample: 32 # Unlikely to run into oom errors since gt is generally small
max_gt_upsample: 4 # Avoid training on excessively upsampled gt
max_raw_training_downsample: 16 # only issue if pyramid doesn't exist, can cause oom errors
max_raw_training_upsample: 2 # avoid training on excessively upsampled raw (probably never an issue)
max_raw_validation_downsample: 8 # only issue if pyramid doesn't exist. Can cause oom errors
max_raw_validation_upsample: 2 # probably never an issue
# for low res models, at what point do we drop crops from datasplit
min_training_volume_size: 8_000 # 20**3

def generate_dataspec_from_csv(csv_path: Path):
    datasets = []
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file {csv_path} does not exist.")
    with open(csv_path, "r") as f:
        for line in f:
            dataset_type,raw_path, gt_path = line.strip().split(",")
            datasets.append(DatasetSpec(
                dataset_type=DatasetType[dataset_type.lower()],
                raw_path=Path(raw_path),
                gt_path=Path(gt_path)
            ))
        
    return datasets


class DatasetSpec:
    dataset_type: DatasetType
    raw_path: Path
    raw_dataset : str
    gt_path: Path
    gt_dataset : str

    def __init__(self, dataset_type: DatasetType, raw_path: Path, raw_dataset: str, gt_path: Path, gt_dataset: str):
        self.dataset_type = dataset_type
        self.raw_path = raw_path
        self.raw_dataset = raw_dataset
        self.gt_path = gt_path
        self.gt_dataset = gt_dataset


class DataSplitGenerator:

    def __init__(self, datasets: List[DatasetSpec]):
        self.datasets = datasets

    # TODO
    def generate(self, task_config: TaskConfig):
        # TODO get instance seg / semantic seg from somewhere
        instance_seg = False
        if instance_seg:
            return self._generate_instance_seg_datasplit(task_config)
        else:
            return self._generate_semantic_seg_datasplit(task_config)
    
    def _generate_instance_seg_datasplit(self, task_config: TaskConfig):
        for dataset in self.datasets:
            labels_config, mask_config, sample_points = self._generate_instance_seg_dataspec(dataset, task_config)
                if labels_config is not None:
                    yield labels_config, mask_config, sample_points

    def _generate_instance_seg_dataspec(self, dataset: DatasetSpec, task_config: TaskConfig):
        raw_path = dataset.raw_path
        raw_dataset = dataset.raw_dataset
        gt_path = dataset.gt_path
        gt_dataset = dataset.gt_dataset

        # TODO
        resolution = task_config.resolution

        if not (raw_path/raw_dataset).exists():
            raise FileNotFoundError(f"Raw path {raw_path/raw_dataset} does not exist.")
        
        if not (gt_path/gt_dataset).exists():
            raise FileNotFoundError(f"GT path {gt_path/gt_dataset} does not exist.")
            

        array_name = f"{gt_path.name}"
        gt_labels_config = ZarrArrayConfig(
                        name=f"gt_{array_name}",
                        file_name=gt_path,
                        dataset=gt_dataset,
                        snap_to_grid=input_voxel_size,
                    )
        labels = gt_labels_config.array_type(gt_labels_config)
        label_voxel_size = labels.voxel_size
        label_roi = labels.roi
        if (
            min_training_volume_size is not None
            and np.prod(label_roi.shape / resolution) < min_training_volume_size
        ):
            logger.debug(
                f"{gt_path/gt_dataset}, shape: {label_roi.shape / resolution} at resolution: {resolution}! Too small"
            )
            return None, None, None

                    upsample = label_voxel_size / resolution
                    downsample = resolution / label_voxel_size
                    if any([u > 1 or d > 1 for u, d in zip(upsample, downsample)]):
                        if skip_large_sampling_factor and any(
                            [x > constants["max_gt_upsample"] for x in upsample]
                            + [
                                x > constants["max_gt_downsample"]
                                for x in downsample
                            ]
                        ):
                            # resampling too extreme. Can lead to memory errors:
                            logger.debug(
                                f"skipping crop {crop_num} due to extreme resampling"
                            )
                            return None, None, None
                        sub_labels_config = ResampledArrayConfig(
                            name=f"{sub_labels_config.name}_resampled_{resolution[0]}nm",
                            source_array_config=sub_labels_config,
                            upsample=upsample,
                            downsample=downsample,
                            interp_order=0,
                        )
                    labels_configs.append(sub_labels_config)
            labels_config = MergeInstancesArrayConfig(
                name=f"{dataset}_{crop_num}_gt",
                source_array_configs=labels_configs,
            )
            mask_config = BinarizeArrayConfig(
                f"{dataset}_{crop_num}_{targets}_{resolution[0]}nm_mask_1",
                groupings=[("nerve_mask", [x for x in range(1, 1000)])],
                source_array_config=labels_config,
            )
        else:
            labels_config = ZarrArrayConfig(
                name=f"{dataset}_{crop_num}_gt",
                file_name=container_path,
                dataset=array_name,
                snap_to_grid=input_voxel_size,
            )

            labels = labels_config.array_type(labels_config)
            label_voxel_size = labels.voxel_size
            label_roi = labels.roi
            if (
                min_size is not None
                and np.prod(label_roi.shape / resolution) < min_size
            ):
                logger.debug(
                    f"{dataset}, {crop_num} shape: {label_roi.shape / resolution} at resolution: {resolution}! Too small"
                )
                return None, None, None

            upsample = label_voxel_size / resolution
            downsample = resolution / label_voxel_size
            if any([u > 1 or d > 1 for u, d in zip(upsample, downsample)]):
                if skip_large_sampling_factor and any(
                    [x > constants["max_gt_upsample"] for x in upsample]
                    + [x > constants["max_gt_downsample"] for x in downsample]
                ):
                    # resampling too extreme. Can lead to memory errors:
                    logger.debug(
                        f"skipping crop {crop_num} due to extreme resampling"
                    )
                    return None, None, None
                labels_config = ResampledArrayConfig(
                    name=f"{labels_config.name}_resampled_{resolution[0]}nm",
                    source_array_config=labels_config,
                    upsample=upsample,
                    downsample=downsample,
                    interp_order=0,
                )
            mask_config = OnesArrayConfig(
                f"{dataset}_{crop_num}_{targets}_{resolution[0]}nm_mask_1",
                source_array_config=labels_config,
            )
        return labels_config, mask_config, sample_points
    else:
        return None, None, None
    

    def _generate_semantic_seg_datasplit(self, task_config: TaskConfig):
        organelle_arrays = {}
        for organelle, _ in targets[1]:
            array_name = f"{labels_group}/{organelle}"
            if Path(container_path, array_name).exists():
                organelle_labels_config = ZarrArrayConfig(
                    f"{dataset}_{crop_num}_{organelle}",
                    container_path,
                    array_name,
                    snap_to_grid=input_voxel_size,
                )

                organelle_labels = ZarrArray(organelle_labels_config)
                if no_useful_data(organelle_labels, targets):
                    logger.debug(
                        f"Skipping {dataset}, {crop_num} due to not containing "
                        f"relevant data for target: {targets[0]}"
                    )
                    continue
                organelle_labels.attrs["labels"]
                label_voxel_size = organelle_labels.voxel_size

                upsample = label_voxel_size / resolution
                downsample = resolution / label_voxel_size
                if any([u > 1 or d > 1 for u, d in zip(upsample, downsample)]):
                    if skip_large_sampling_factor and any(
                        [x > constants["max_gt_upsample"] for x in upsample]
                        + [x > constants["max_gt_downsample"] for x in downsample]
                    ):
                        # resampling too extreme. Can lead to memory errors:
                        logger.debug(
                            f"skipping crop {crop_num} due to extreme resampling"
                        )
                        return None, None, None
                    organelle_labels_config = ResampledArrayConfig(
                        name=f"{organelle_labels_config.name}_resampled_{resolution[0]}nm",
                        source_array_config=organelle_labels_config,
                        upsample=upsample,
                        downsample=downsample,
                        interp_order=0,
                    )

                # binarize everything into 0 or 1
                organelle_gt_config = BinarizeArrayConfig(
                    f"{dataset}_{crop_num}_{organelle}_{resolution[0]}nm_binarized",
                    source_array_config=organelle_labels_config,
                    groupings=[(organelle, [])],
                )
                # mask in everything in this array
                organelle_mask_config = OnesArrayConfig(
                    f"{dataset}_{crop_num}_{organelle}_{resolution[0]}nm_mask_1",
                    source_array_config=organelle_gt_config,
                )
                organelle_arrays[organelle] = (
                    organelle_gt_config,
                    organelle_mask_config,
                )

        # Concatenates multiple arrays along the channel dimension, giving each channel
        # the name of the organelle from which it came. If no array is provided for
        # a channel it will be filled with zeros

        # Assume mutual exclusivity. e.g. nucleus cannot also be mito. So although
        # mitos may not be annotated in a nucleus crop, we can at least train the
        # negative case wherever there is nucleus.
        gt_config = ConcatArrayConfig(
            name=f"{dataset}_{crop_num}_{resolution[0]}nm_gt",
            channels=[organelle for organelle, _ in targets[1]],
            source_array_configs={k: gt for k, (gt, _) in organelle_arrays.items()},
        )
        label_mask_config = LogicalOrArrayConfig(
            name=f"{dataset}_{crop_num}_{resolution[0]}nm_labelled_voxels",
            source_array_config=gt_config,
        )
        mask_config = ConcatArrayConfig(
            name=f"{dataset}_{crop_num}_{resolution[0]}nm_mask",
            channels=[organelle for organelle, _ in targets[1]],
            source_array_configs={k: mask for k, (_, mask) in organelle_arrays.items()},
            default_config=label_mask_config,
        )
        if (
            len(gt_config.source_array_configs) == 0
            or len(mask_config.source_array_configs) == 0
        ):
            return None, None, None
        else:
            return gt_config, mask_config, None
        return datasplit
    
    def generate_csv(self, csv_path: Path):
        print(f"Writing dataspecs to {csv_path}")
        with open(csv_path, "w") as f:
            for dataset in self.datasets:
                f.write(f"{dataset.dataset_type.name},{dataset.raw_path},{dataset.gt_path}\n")

    @staticmethod
    def generate_from_csv(csv_path: Path):
        return DataSplitGenerator(generate_dataspec_from_csv(csv_path))


# datasplit = DataSplitGenerator.generate_from_csv(Path("")).generate(
#     task_config=task_config
# )



# create test csv
csv_path = Path("/groups/cellmap/cellmap/zouinkhim/refactor/dacapo_11/dacapo/dacapo/experiments/datasplits/test.csv")
datasets = [
    DatasetSpec(
        dataset_type=DatasetType.val,
        raw_path=Path("/groups/funke/funkelab/projects/livseg/data/training_data/moa1.zarr/raw"),
        gt_path=Path("/groups/funke/funkelab/projects/livseg/data/training_data/moa1.zarr/label")
    ),
    DatasetSpec(
        dataset_type=DatasetType.train,
        raw_path=Path("/groups/funke/funkelab/projects/livseg/data/training_data/cc1.zarr/raw"),
        gt_path=Path("/groups/funke/funkelab/projects/livseg/data/training_data/cc1.zarr/label")
    )
]
DataSplitGenerator(datasets).generate_csv(csv_path)