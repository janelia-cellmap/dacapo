# Dacapo

## Imports


```python
from pathlib import PosixPath
from dacapo.experiments.datasplits.datasets.arrays import (
    BinarizeArrayConfig,
    IntensitiesArrayConfig,
    MissingAnnotationsMaskConfig,
    ResampledArrayConfig,
    ZarrArrayConfig,
)
from dacapo.experiments.tasks import DistanceTaskConfig
from dacapo.experiments.architectures import CNNectomeUNetConfig
from dacapo.experiments.trainers import GunpowderTrainerConfig
from dacapo.experiments.trainers.gp_augments import (
    ElasticAugmentConfig,
    GammaAugmentConfig,
    IntensityAugmentConfig,
    IntensityScaleShiftAugmentConfig,
)
from dacapo.experiments.datasplits import TrainValidateDataSplitConfig
from dacapo.experiments.datasplits.datasets import RawGTDatasetConfig
from dacapo.experiments.starts import StartConfig
from dacapo.experiments import RunConfig
from dacapo.store.create_store import create_config_store
```

## Config Store


```python
config_store = create_config_store()
```

## Task


```python
task_config = DistanceTaskConfig(
    name="example_distances_8nm_peroxisome",
    channels=["peroxisome"],
    clip_distance=80.0,
    tol_distance=80.0,
    scale_factor=160.0,
    mask_distances=True,
)
config_store.store_task_config(task_config)
```

## Architecture


```python
architecture_config = CNNectomeUNetConfig(
    name="example_attention-upsample-unet",
    input_shape=(216, 216, 216),
    fmaps_out=72,
    fmaps_in=1,
    num_fmaps=12,
    fmap_inc_factor=6,
    downsample_factors=[(2, 2, 2), (3, 3, 3), (3, 3, 3)],
    kernel_size_down=None,
    kernel_size_up=None,
    eval_shape_increase=(72, 72, 72),
    upsample_factors=[(2, 2, 2)],
    constant_upsample=True,
    padding="valid",
)
config_store.store_architecture_config(architecture_config)
```

## Trainer


```python
trainer_config = GunpowderTrainerConfig(
    name="example_default_one_label_finetuning",
    batch_size=2,
    learning_rate=1e-05,
    num_data_fetchers=20,
    augments=[
        ElasticAugmentConfig(
            control_point_spacing=[100, 100, 100],
            control_point_displacement_sigma=[10.0, 10.0, 10.0],
            rotation_interval=(0.0, 1.5707963267948966),
            subsample=8,
            uniform_3d_rotation=True,
        ),
        IntensityAugmentConfig(scale=(0.5, 1.5), shift=(-0.2, 0.2), clip=True),
        GammaAugmentConfig(gamma_range=(0.5, 1.5)),
        IntensityScaleShiftAugmentConfig(scale=2.0, shift=-1.0),
    ],
    snapshot_interval=10000,
    min_masked=0.05,
    clip_raw=False,
)
config_store.store_trainer_config(trainer_config)
```

## Datasplit


```python
datasplit_config = TrainValidateDataSplitConfig(
    name="example_jrc_mus-livers_peroxisome_8nm",
    train_configs=[
        RawGTDatasetConfig(
            name="jrc_mus-liver_124_peroxisome_8nm",
            weight=1,
            raw_config=IntensitiesArrayConfig(
                name="jrc_mus-liver_s1_raw",
                source_array_config=ZarrArrayConfig(
                    name="jrc_mus-liver_raw_uint8",
                    file_name=PosixPath(
                        "/nrs/cellmap/data/jrc_mus-liver/jrc_mus-liver.n5"
                    ),
                    dataset="volumes/raw/s1",
                    snap_to_grid=(16, 16, 16),
                    axes=None,
                ),
                min=0.0,
                max=255.0,
            ),
            gt_config=BinarizeArrayConfig(
                name="jrc_mus-liver_124_peroxisome_8nm_gt",
                source_array_config=ResampledArrayConfig(
                    name="jrc_mus-liver_124_gt_resampled_8nm",
                    source_array_config=ZarrArrayConfig(
                        name="jrc_mus-liver_124_gt",
                        file_name=PosixPath(
                            "/nrs/cellmap/zouinkhim/data/tmp_data_v3/jrc_mus-liver/jrc_mus-liver.n5"
                        ),
                        dataset="volumes/groundtruth/crop124/labels//all",
                        snap_to_grid=(16, 16, 16),
                        axes=None,
                    ),
                    upsample=(0, 0, 0),
                    downsample=(2, 2, 2),
                    interp_order=False,
                ),
                groupings=[("peroxisome", [47, 48])],
                background=0,
            ),
            mask_config=MissingAnnotationsMaskConfig(
                name="jrc_mus-liver_124_peroxisome_8nm_mask",
                source_array_config=ResampledArrayConfig(
                    name="jrc_mus-liver_124_gt_resampled_8nm",
                    source_array_config=ZarrArrayConfig(
                        name="jrc_mus-liver_124_gt",
                        file_name=PosixPath(
                            "/nrs/cellmap/zouinkhim/data/tmp_data_v3/jrc_mus-liver/jrc_mus-liver.n5"
                        ),
                        dataset="volumes/groundtruth/crop124/labels//all",
                        snap_to_grid=(16, 16, 16),
                        axes=None,
                    ),
                    upsample=(0, 0, 0),
                    downsample=(2, 2, 2),
                    interp_order=False,
                ),
                groupings=[("peroxisome", [47, 48])],
            ),
            sample_points=None,
        ),
        RawGTDatasetConfig(
            name="jrc_mus-liver_125_peroxisome_8nm",
            weight=1,
            raw_config=IntensitiesArrayConfig(
                name="jrc_mus-liver_s1_raw",
                source_array_config=ZarrArrayConfig(
                    name="jrc_mus-liver_raw_uint8",
                    file_name=PosixPath(
                        "/nrs/cellmap/data/jrc_mus-liver/jrc_mus-liver.n5"
                    ),
                    dataset="volumes/raw/s1",
                    snap_to_grid=(16, 16, 16),
                    axes=None,
                ),
                min=0.0,
                max=255.0,
            ),
            gt_config=BinarizeArrayConfig(
                name="jrc_mus-liver_125_peroxisome_8nm_gt",
                source_array_config=ResampledArrayConfig(
                    name="jrc_mus-liver_125_gt_resampled_8nm",
                    source_array_config=ZarrArrayConfig(
                        name="jrc_mus-liver_125_gt",
                        file_name=PosixPath(
                            "/nrs/cellmap/zouinkhim/data/tmp_data_v3/jrc_mus-liver/jrc_mus-liver.n5"
                        ),
                        dataset="volumes/groundtruth/crop125/labels//all",
                        snap_to_grid=(16, 16, 16),
                        axes=None,
                    ),
                    upsample=(0, 0, 0),
                    downsample=(2, 2, 2),
                    interp_order=False,
                ),
                groupings=[("peroxisome", [47, 48])],
                background=0,
            ),
            mask_config=MissingAnnotationsMaskConfig(
                name="jrc_mus-liver_125_peroxisome_8nm_mask",
                source_array_config=ResampledArrayConfig(
                    name="jrc_mus-liver_125_gt_resampled_8nm",
                    source_array_config=ZarrArrayConfig(
                        name="jrc_mus-liver_125_gt",
                        file_name=PosixPath(
                            "/nrs/cellmap/zouinkhim/data/tmp_data_v3/jrc_mus-liver/jrc_mus-liver.n5"
                        ),
                        dataset="volumes/groundtruth/crop125/labels//all",
                        snap_to_grid=(16, 16, 16),
                        axes=None,
                    ),
                    upsample=(0, 0, 0),
                    downsample=(2, 2, 2),
                    interp_order=False,
                ),
                groupings=[("peroxisome", [47, 48])],
            ),
            sample_points=None,
        ),
    ],
    validate_configs=[
        RawGTDatasetConfig(
            name="jrc_mus-liver_145_peroxisome_8nm",
            weight=1,
            raw_config=IntensitiesArrayConfig(
                name="jrc_mus-liver_s1_raw",
                source_array_config=ZarrArrayConfig(
                    name="jrc_mus-liver_raw_uint8",
                    file_name=PosixPath(
                        "/nrs/cellmap/data/jrc_mus-liver/jrc_mus-liver.n5"
                    ),
                    dataset="volumes/raw/s1",
                    snap_to_grid=(16, 16, 16),
                    axes=None,
                ),
                min=0.0,
                max=255.0,
            ),
            gt_config=BinarizeArrayConfig(
                name="jrc_mus-liver_145_peroxisome_8nm_gt",
                source_array_config=ResampledArrayConfig(
                    name="jrc_mus-liver_145_gt_resampled_8nm",
                    source_array_config=ZarrArrayConfig(
                        name="jrc_mus-liver_145_gt",
                        file_name=PosixPath(
                            "/nrs/cellmap/zouinkhim/data/tmp_data_v3/jrc_mus-liver/jrc_mus-liver.n5"
                        ),
                        dataset="volumes/groundtruth/crop145/labels//all",
                        snap_to_grid=(16, 16, 16),
                        axes=None,
                    ),
                    upsample=(0, 0, 0),
                    downsample=(2, 2, 2),
                    interp_order=False,
                ),
                groupings=[("peroxisome", [47, 48])],
                background=0,
            ),
            mask_config=MissingAnnotationsMaskConfig(
                name="jrc_mus-liver_145_peroxisome_8nm_mask",
                source_array_config=ResampledArrayConfig(
                    name="jrc_mus-liver_145_gt_resampled_8nm",
                    source_array_config=ZarrArrayConfig(
                        name="jrc_mus-liver_145_gt",
                        file_name=PosixPath(
                            "/nrs/cellmap/zouinkhim/data/tmp_data_v3/jrc_mus-liver/jrc_mus-liver.n5"
                        ),
                        dataset="volumes/groundtruth/crop145/labels//all",
                        snap_to_grid=(16, 16, 16),
                        axes=None,
                    ),
                    upsample=(0, 0, 0),
                    downsample=(2, 2, 2),
                    interp_order=False,
                ),
                groupings=[("peroxisome", [47, 48])],
            ),
            sample_points=None,
        )
    ],
)

config_store.store_datasplit_config(datasplit_config)
```

## Run


```python
start_config = StartConfig(
    "setup04",
    "best",
)
iterations = 200000
validation_interval = 5000
repetitions = 3
for i in range(repetitions):
    run_config = RunConfig(
        name=("_").join(
            [
                "example",
                "scratch" if start_config is None else "finetuned",
                task_config.name,
                architecture_config.name,
                trainer_config.name,
                datasplit_config.name,
            ]
        )
        + f"__{i}",
        task_config=task_config,
        architecture_config=architecture_config,
        trainer_config=trainer_config,
        datasplit_config=datasplit_config,
        num_iterations=iterations,
        validation_interval=validation_interval,
        repetition=i,
        start_config=start_config,
    )

    print(run_config.name)
    config_store.store_run_config(run_config)
```

    example_finetuned_example_distances_8nm_peroxisome_example_attention-upsample-unet_example_default_one_label_finetuning_example_jrc_mus-livers_peroxisome_8nm__0
    example_finetuned_example_distances_8nm_peroxisome_example_attention-upsample-unet_example_default_one_label_finetuning_example_jrc_mus-livers_peroxisome_8nm__1
    example_finetuned_example_distances_8nm_peroxisome_example_attention-upsample-unet_example_default_one_label_finetuning_example_jrc_mus-livers_peroxisome_8nm__2

