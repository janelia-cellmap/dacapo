# Dacapo

DaCapo is a framework that allows for easy configuration and execution of established machine learning techniques on arbitrarily large volumes of multi-dimensional images.

DaCapo has 4 major configurable components:
1. **dacapo.datasplits.DataSplit**

2. **dacapo.tasks.Task**

3. **dacapo.architectures.Architecture**

4. **dacapo.trainers.Trainer**

These are then combined in a single **dacapo.experiments.Run** that includes your starting point (whether you want to start training from scratch or continue off of a previously trained model) and stopping criterion (the number of iterations you want to train).

## Environment setup
If you have not already done so, you will need to install DaCapo. We recommend you do this by first creating a new environment and then installing DaCapo using pip.

```bash
conda create -n dacapo python=3.10
conda activate dacapo
```

Then, you can install DaCapo using pip, via GitHub:

```bash
pip install git+https://github.com/janelia-cellmap/dacapo.git
```

Or you can clone the repository and install it locally:

```bash
git clone https://github.com/janelia-cellmap/dacapo.git
cd dacapo
pip install -e .
```


## Config Store

To define where the data goes, create a dacapo.yaml configuration file. Here is a template:
```yaml 
mongodbhost: mongodb://dbuser:dbpass@dburl:dbport/
mongodbname: dacapo
runs_base_dir: /path/to/my/data/storage
```
The `runs_base_dir` defines where your on-disk data will be stored. The `mongodbhost` and `mongodbname` define the mongodb host and database that will store your cloud data. If you want to store everything on disk, replace `mongodbhost` and `mongodbname` with a single type: files and everything will be saved to disk.


```python
from dacapo.store.create_store import create_config_store

config_store = create_config_store()
```

## Datasplit
Where can you find your data? What format is it in? Does it need to be normalized? What data do you want to use for validation?


```python
from dacapo.experiments.datasplits.datasets.arrays import (
    BinarizeArrayConfig,
    IntensitiesArrayConfig,
    MissingAnnotationsMaskConfig,
    ResampledArrayConfig,
    ZarrArrayConfig,
)
from dacapo.experiments.datasplits import TrainValidateDataSplitConfig
from dacapo.experiments.datasplits.datasets import RawGTDatasetConfig
from pathlib import PosixPath

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

## Task
What do you want to learn? An instance segmentation? If so, how? Affinities,
Distance Transform, Foreground/Background, etc. Each of these tasks are commonly learned
and evaluated with specific loss functions and evaluation metrics. Some tasks may
also require specific non-linearities or output formats from your model.


```python
from dacapo.experiments.tasks import DistanceTaskConfig

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

The setup of the network you will train. Biomedical image to image translation often utilizes a UNet, but even after choosing a UNet you still need to provide some additional parameters. How much do you want to downsample? How many convolutional layers do you want?


```python
from dacapo.experiments.architectures import CNNectomeUNetConfig

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

How do you want to train? This config defines the training loop and how the other three components work together. What sort of augmentations to apply during training, what learning rate and optimizer to use, what batch size to train with.


```python
from dacapo.experiments.trainers import GunpowderTrainerConfig
from dacapo.experiments.trainers.gp_augments import (
    ElasticAugmentConfig,
    GammaAugmentConfig,
    IntensityAugmentConfig,
    IntensityScaleShiftAugmentConfig,
)

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

## Run
Now that we have our components configured, we just need to combine them into a run and start training. We can have multiple repetitions of a single set of configs in order to increase our chances of finding an optimum.


```python
from dacapo.experiments.starts import StartConfig
from dacapo.experiments import RunConfig
from dacapo.experiments.run import Run

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
                datasplit_config.name,
                task_config.name,
                architecture_config.name,
                trainer_config.name,
            ]
        )
        + f"__{i}",
        datasplit_config=datasplit_config,
        task_config=task_config,
        architecture_config=architecture_config,
        trainer_config=trainer_config,
        num_iterations=iterations,
        validation_interval=validation_interval,
        repetition=i,
        start_config=start_config,
    )

    print(run_config.name)
    config_store.store_run_config(run_config)
```

    example_finetuned_example_jrc_mus-livers_peroxisome_8nm_example_distances_8nm_peroxisome_example_attention-upsample-unet_example_default_one_label_finetuning__0
    example_finetuned_example_jrc_mus-livers_peroxisome_8nm_example_distances_8nm_peroxisome_example_attention-upsample-unet_example_default_one_label_finetuning__1
    example_finetuned_example_jrc_mus-livers_peroxisome_8nm_example_distances_8nm_peroxisome_example_attention-upsample-unet_example_default_one_label_finetuning__2


## Train

To train one of the runs, you can either do it by first creating a **Run** directly from the run config


```python
from dacapo.train import train_run

run = Run(run_config)
train_run(run)
```

Or - since we already stored the configs - we can start the run via just the run name:


```python
train_run(run_config.name)
```

If you want to start your run on some compute cluster, you might want to use the command line interface: dacapo train -r {run_config.name}. This makes it particularly convenient to run on compute nodes where you can specify specific compute requirements.
