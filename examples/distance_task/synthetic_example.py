# %% [markdown]
# # Dacapo
#
# DaCapo is a framework that allows for easy configuration and execution of established machine learning techniques on arbitrarily large volumes of multi-dimensional images.
#
# DaCapo has 4 major configurable components:
# 1. **dacapo.datasplits.DataSplit**
#
# 2. **dacapo.tasks.Task**
#
# 3. **dacapo.architectures.Architecture**
#
# 4. **dacapo.trainers.Trainer**
#
# These are then combined in a single **dacapo.experiments.Run** that includes your starting point (whether you want to start training from scratch or continue off of a previously trained model) and stopping criterion (the number of iterations you want to train).

# %% [markdown]
# ## Environment setup
# If you have not already done so, you will need to install DaCapo. You can do this by first creating a new environment and then installing DaCapo using pip.
#
# ```bash
# conda create -n dacapo python=3.10
# conda activate dacapo
# ```
#
# Then, you can install DaCapo using pip, via GitHub:
#
# ```bash
# pip install git+https://github.com/janelia-cellmap/dacapo.git
# ```
#
# Or you can clone the repository and install it locally:
#
# ```bash
# git clone https://github.com/janelia-cellmap/dacapo.git
# cd dacapo
# pip install -e .
# ```
#
# Be sure to select this environment in your Jupyter notebook or JupyterLab.

# %% [markdown]
"""
## Config Store
To define where the data goes, create a dacapo.yaml configuration file either in `~/.config/dacapo/dacapo.yaml` or in `./dacapo.yaml`. Here is a template:
```yaml
type: files
runs_base_dir: /path/to/my/data/storage
```
The `runs_base_dir` defines where your on-disk data will be stored. The `type` setting determines the database backend. The default is `files`, which stores the data in a file tree on disk. Alternatively, you can use `mongodb` to store the data in a MongoDB database. To use MongoDB, you will need to provide a `mongodbhost` and `mongodbname` in the configuration file:

```yaml
...
mongodbhost: mongodb://dbuser:dbpass@dburl:dbport/
mongodbname: dacapo
"""
# %%
# First we need to create a config store to store our configurations
from dacapo.store.create_store import create_config_store

config_store = create_config_store()


# %%
# Then let's make sure we have data to train on. If this is already provided, you can skip to the Datasplit section.
# %%
from pathlib import Path
from dacapo import Options
from dacapo.utils.view import get_viewer
from examples.synthetic_source_worker import generate_synthetic_dataset
from funlib.geometry import Coordinate
from funlib.persistence import open_ds

options = Options.instance()
runs_base_dir = options.runs_base_dir
force_example_creation = False
num_workers = 32

# First for training data
train_data_path = Path(runs_base_dir, "example_train.zarr")
try:
    assert not force_example_creation
    raw_array = open_ds(str(train_data_path), "raw")
    labels_array = open_ds(str(train_data_path), "labels")
except:
    train_shape = Coordinate((512, 512, 512))
    generate_synthetic_dataset(
        train_data_path,
        shape=train_shape,
        overwrite=True,
        num_workers=num_workers,
        write_shape=Coordinate((128, 128, 128)),
    )
    raw_array = open_ds(str(train_data_path), "raw")
    labels_array = open_ds(str(train_data_path), "labels")
arrays = {
    "raw": {"array": raw_array},
    "labels": {"array": labels_array, "meshes": True},
}
get_viewer(arrays, headless=False)

# %%
# Then for validation data
validate_data_path = Path(runs_base_dir, "example_validate.zarr")
try:
    assert not force_example_creation
    raw_array = open_ds(str(validate_data_path), "raw")
    labels_array = open_ds(str(validate_data_path), "labels")
except:
    validate_shape = Coordinate((152, 152, 152)) * 1
    generate_synthetic_dataset(
        validate_data_path,
        shape=validate_shape,
        write_shape=Coordinate((152, 152, 152)),
        overwrite=True,
        num_workers=num_workers,
    )

arrays = {
    "raw": {"array": raw_array},
    "labels": {"array": labels_array, "meshes": True},
}
get_viewer(arrays, headless=False)

# %%
# Then let's make some test data
test_data_path = Path(runs_base_dir, "example_test.zarr")
try:
    assert not force_example_creation
    raw_array = open_ds(str(test_data_path), "raw")
    labels_array = open_ds(str(test_data_path), "labels")
except:
    test_shape = Coordinate((152, 152, 152)) * 3
    generate_synthetic_dataset(
        test_data_path,
        shape=test_shape,
        overwrite=True,
        write_shape=Coordinate((152, 152, 152)),
        num_workers=num_workers,
    )

arrays = {
    "raw": {"array": raw_array},
    "labels": {"array": labels_array, "meshes": True},
}
get_viewer(arrays, headless=False)

# %% [markdown]
# ## Datasplit
# Where can you find your data? What format is it in? Does it need to be normalized? What data do you want to use for validation?

# We'll assume your data is in a zarr file, and that you have a raw and a ground truth dataset, all stored in your `runs_base_dir` as `example_{type}.zarr` where `{type}` is either `train` or `validate`.
# NOTE: You may need to delete old config stores if you are re-running this cell with modifications to the configs. The config names are unique and will throw an error if you try to store a config with the same name as an existing config. For the `files` backend, you can delete the `runs_base_dir/configs` directory to remove all stored configs.

# %%
from pathlib import Path
from dacapo.experiments.datasplits import DataSplitGenerator
from funlib.geometry import Coordinate

csv_path = Path(runs_base_dir, "synthetic_example.csv")
if not csv_path.exists():
    # Create a csv file with the paths to the zarr files
    with open(csv_path, "w") as f:
        f.write(
            f"train,{train_data_path},raw,{train_data_path},[labels]\n"
            f"val,{validate_data_path},raw,{validate_data_path},[labels]\n"
            # f"test,{test_data_path},raw,{test_data_path},[labels]\n"
        )

input_resolution = Coordinate(8, 8, 8)
output_resolution = Coordinate(8, 8, 8)
datasplit_config = DataSplitGenerator.generate_from_csv(
    csv_path,
    input_resolution,
    output_resolution,
    binarize_gt=True,  # Binarize the ground truth data to convert from instance segmentation to semantic segmentation
).compute()

datasplit = datasplit_config.datasplit_type(datasplit_config)
viewer = datasplit._neuroglancer()
# config_store.store_datasplit_config(datasplit_config)

# %% [markdown]
# The above datasplit_generator automates a lot of the heavy lifting for configuring data to set up a run. The following shows everything that it is doing, and an equivalent way to set up the datasplit.
# ```python
# datasplit_config = TrainValidateDataSplitConfig(
#     name="synthetic_example_semantic_['labels']_8nm",
#     train_configs=[
#         RawGTDatasetConfig(
#             name="example_train_[labels]_['labels']_8nm",
#             weight=1,
#             raw_config=IntensitiesArrayConfig(
#                 name="raw_example_train_uint8",
#                 source_array_config=ZarrArrayConfig(
#                     name="raw_example_train_uint8",
#                     file_name=Path(
#                         "/misc/public/dacapo_learnathon/synthetic/example_train.zarr"
#                     ),
#                     dataset="raw",
#                 ),
#                 min=0,
#                 max=255,
#             ),
#             gt_config=BinarizeArrayConfig(
#                 name="example_train_[labels]_labels_8nm_binarized",
#                 source_array_config=ZarrArrayConfig(
#                     name="gt_example_train_labels_uint8",
#                     file_name=Path(
#                         "/misc/public/dacapo_learnathon/synthetic/example_train.zarr"
#                     ),
#                     dataset="labels",
#                 ),
#                 groupings=[("labels", [])],
#                 background=0,
#             ),
#             mask_config=None,
#             sample_points=None,
#         )
#     ],
#     validate_configs=[
#         RawGTDatasetConfig(
#             name="example_validate_[labels]_['labels']_8nm",
#             weight=1,
#             raw_config=IntensitiesArrayConfig(
#                 name="raw_example_validate_uint8",
#                 source_array_config=ZarrArrayConfig(
#                     name="raw_example_validate_uint8",
#                     file_name=Path(
#                         "/misc/public/dacapo_learnathon/synthetic/example_validate.zarr"
#                     ),
#                     dataset="raw",
#                 ),
#                 min=0,
#                 max=255,
#             ),
#             gt_config=BinarizeArrayConfig(
#                 name="example_validate_[labels]_labels_8nm_binarized",
#                 source_array_config=ZarrArrayConfig(
#                     name="gt_example_validate_labels_uint8",
#                     file_name=Path(
#                         "/misc/public/dacapo_learnathon/synthetic/example_validate.zarr"
#                     ),
#                     dataset="labels",
#                 ),
#                 groupings=[("labels", [])],
#                 background=0,
#             ),
#             mask_config=None,
#             sample_points=None,
#         )
#     ],
# )
# config_store.store_datasplit_config(datasplit_config)
# datasplit = datasplit_config.datasplit_type(datasplit_config)
# viewer = datasplit._neuroglancer()
# ```


# %% [markdown]
# ## Task
# What do you want to learn? An instance segmentation? If so, how? Affinities,
# Distance Transform, Foreground/Background, etc. Each of these tasks are commonly learned
# and evaluated with specific loss functions and evaluation metrics. Some tasks may
# also require specific non-linearities or output formats from your model.

# %%
from dacapo.experiments.tasks import DistanceTaskConfig

task_config = DistanceTaskConfig(
    name="example_distance_task",
    channels=["labels"],
    clip_distance=80.0,
    tol_distance=80.0,
    scale_factor=160.0,
)
# config_store.store_task_config(task_config)

# %% [markdown]
# ## Architecture
#
# The setup of the network you will train. Biomedical image to image translation often utilizes a UNet, but even after choosing a UNet you still need to provide some additional parameters. How much do you want to downsample? How many convolutional layers do you want?

# %%
from dacapo.experiments.architectures import CNNectomeUNetConfig

architecture_config = CNNectomeUNetConfig(
    name="example-mini_unet",
    input_shape=(172, 172, 172),
    fmaps_out=24,
    fmaps_in=1,
    num_fmaps=12,
    fmap_inc_factor=2,
    downsample_factors=[(2, 2, 2), (2, 2, 2), (2, 2, 2)],
    eval_shape_increase=(72, 72, 72),
)
# try:
#     config_store.store_architecture_config(architecture_config)
# except:
#     config_store.delete_architecture_config(architecture_config.name)
#     config_store.store_architecture_config(architecture_config)

# %% [markdown]
# ## Trainer
#
# How do you want to train? This config defines the training loop and how the other three components work together. What sort of augmentations to apply during training, what learning rate and optimizer to use, what batch size to train with.

# %%
from dacapo.experiments.trainers import GunpowderTrainerConfig
from dacapo.experiments.trainers.gp_augments import (
    ElasticAugmentConfig,
    GammaAugmentConfig,
    IntensityAugmentConfig,
    IntensityScaleShiftAugmentConfig,
)

trainer_config = GunpowderTrainerConfig(
    name="synthetic_distance_trainer",
    batch_size=1,
    learning_rate=0.0001,
    num_data_fetchers=20,
    augments=[
        ElasticAugmentConfig(
            control_point_spacing=[100, 100, 100],
            control_point_displacement_sigma=[10.0, 10.0, 10.0],
            rotation_interval=(0.0, 1.5707963267948966),
            subsample=8,
            uniform_3d_rotation=True,
        ),
        IntensityAugmentConfig(scale=(0.25, 1.75), shift=(-0.5, 0.35), clip=True),
        GammaAugmentConfig(gamma_range=(0.5, 2.0)),
        IntensityScaleShiftAugmentConfig(scale=2.0, shift=-1.0),
    ],
    snapshot_interval=10000,
    min_masked=0.05,
    clip_raw=False,
)
# config_store.store_trainer_config(trainer_config)

# %% [markdown]
# ## Run
# Now that we have our components configured, we just need to combine them into a run and start training. We can have multiple repetitions of a single set of configs in order to increase our chances of finding an optimum.

# %%
from dacapo.experiments import RunConfig
from dacapo.experiments.run import Run

start_config = None

# Uncomment to start from a pretrained model
# start_config = StartConfig(
#     "setup04",
#     "best",
# )

iterations = 2000
validation_interval = 200
repetitions = 1
for i in range(repetitions):
    run_config = RunConfig(
        name="example_synthetic_distance_run",
        # # NOTE: This is a template for the name of the run. You can customize it as you see fit.
        # name=("_").join(
        #     [
        #         "example",
        #         "scratch" if start_config is None else "finetuned",
        #         datasplit_config.name,
        #         task_config.name,
        #         architecture_config.name,
        #         trainer_config.name,
        #     ]
        # )
        # + f"__{i}",
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
    try:
        config_store.store_run_config(run_config)
    except:
        config_store.delete_run_config(run_config.name)
        config_store.store_run_config(run_config)


# %% [markdown]
# ## Train

# NOTE: The run stats are stored in the `runs_base_dir/stats` directory. You can delete this directory to remove all stored stats if you want to re-run training. Otherwise, the stats will be appended to the existing files, and the run won't start from scratch. This may cause errors
# %%
from dacapo.train import train_run
from dacapo.experiments.run import Run
from dacapo.store.create_store import create_config_store
from dacapo.utils.view import NeuroglancerRunViewer

config_store = create_config_store()
run = Run(config_store.retrieve_run_config(run_config.name))

# First visualize all the steps in the data preprocessing pipeline
from dacapo.store.create_store import create_array_store

array_store = create_array_store()
run.trainer.build_batch_provider(
    run.datasplit.train,
    run.model,
    run.task,
    array_store.snapshot_container(run.name),
)
run.trainer.visualize_pipeline()

# %% Now let's train!

# Visualize as we go
run_viewer = NeuroglancerRunViewer(run)
run_viewer.start()
# %%
# Train the run
train_run(run)
# %%
# Stop the viewer
run_viewer.stop()
# %% [markdown]
# If you want to start your run on some compute cluster, you might want to use the command line interface: dacapo train -r {run_config.name}. This makes it particularly convenient to run on compute nodes where you can specify specific compute requirements.


# %% [markdown]
# ## Validate

# Once you have trained your model, you can validate it on the validation datasets used during training. You can use the `dacapo.validate` function to do this. You can also use the command line interface to validate a run: dacapo validate -r {run_config.name} -i {iteration}

# Generally we setup training to automatically validate at a set interval and the model checkpoints are saved at these intervals.

# %%
from dacapo.validate import validate

validate(run_config.name, iterations, num_workers=1, overwrite=True)

# %% [markdown]
# ## Predict
# Once you have trained and validated your model, you can use it to predict on new data. You can use the `dacapo.predict` function to do this. You can also use the command line interface to predict on a run: dacapo predict -r {run_config.name} -i {iteration} -ic {input_container} -id {input_dataset} -op {output_path}

# %%
from dacapo.predict import predict

predict(
    run_config.name,
    iterations,
    test_data_path,
    "raw",
    test_data_path,
    # num_workers=32,
    num_workers=1,
    overwrite=True,
    output_dtype="float32",
    output_roi=raw_array.roi,
)

raw_array = open_ds(str(test_data_path), "raw")
pred_array = open_ds(str(test_data_path), "predictions")
gt_array = open_ds(str(test_data_path), "labels")

arrays = {
    "raw": {"array": raw_array},
    "labels": {"array": gt_array, "meshes": True},
    "predictions": {"array": pred_array},
}
get_viewer(arrays, headless=False)
