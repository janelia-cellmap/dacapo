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
# ## Config Store
# To define where the data goes, create a dacapo.yaml configuration file either in `~/.config/dacapo/dacapo.yaml` or in `./dacapo.yaml`. Here is a template:
#
# ```yaml
# mongodbhost: mongodb://dbuser:dbpass@dburl:dbport/
# mongodbname: dacapo
# runs_base_dir: /path/to/my/data/storage
# ```
# The runs_base_dir defines where your on-disk data will be stored. The mongodbhost and mongodbname define the mongodb host and database that will store your cloud data. If you want to store everything on disk, replace mongodbhost and mongodbname with a single type `files` and everything will be saved to disk:
#
# ```yaml
# type: files
# runs_base_dir: /path/to/my/data/storage
# ```
#


# %%

# for training:

# pipeline, request = random_source_pipeline(input_shape=(512, 512, 512))
# def batch_generator():
#     with gp.build(pipeline):
#         while True:
#             yield pipeline.request_batch(request)
# batch_gen = batch_generator()
# training_batch = next(batch_gen)


# for validation:
# pipeline, request = random_source_pipeline(input_shape=(108, 108, 108))


# def batch_generator():
#     with gp.build(pipeline):
#         while True:
#             yield pipeline.request_batch(request)


# batch_gen = batch_generator()
# validation_batch = next(batch_gen)


# def view_batch(batch):
#     raw_array = batch.arrays[gp.ArrayKey("RAW")]
#     labels_array = batch.arrays[gp.ArrayKey("LABELS")]

#     labels_data = labels_array.data
#     labels_spec = labels_array.spec

#     raw_data = raw_array.data
#     raw_spec = raw_array.spec

#     neuroglancer.set_server_bind_address("0.0.0.0")
#     viewer = neuroglancer.Viewer()
#     with viewer.txn() as state:
#         state.showSlices = False
#         state.layers["segs"] = neuroglancer.SegmentationLayer(
#             # segments=[str(i) for i in np.unique(data[data > 0])], # this line will cause all objects to be selected and thus all meshes to be generated...will be slow if lots of high res meshes
#             source=neuroglancer.LocalVolume(
#                 data=labels_data,
#                 dimensions=neuroglancer.CoordinateSpace(
#                     names=["z", "y", "x"],
#                     units=["nm", "nm", "nm"],
#                     scales=labels_spec.voxel_size,
#                 ),
#                 # voxel_offset=ds.roi.begin / ds.voxel_size,
#             ),
#             segments=np.unique(labels_data[labels_data > 0]),
#         )

#         state.layers["raw"] = neuroglancer.ImageLayer(
#             source=neuroglancer.LocalVolume(
#                 data=raw_data,
#                 dimensions=neuroglancer.CoordinateSpace(
#                     names=["z", "y", "x"],
#                     units=["nm", "nm", "nm"],
#                     scales=raw_spec.voxel_size,
#                 ),
#             ),
#         )
#     return IFrame(src=viewer, width=1500, height=600)


# training_batch

# # %%
# view_batch(training_batch)
# view_batch(validation_batch)
# %%
# write out files
# from funlib.persistence import prepare_ds
# from pathlib import Path

# # Create a temporary directory


# # with tempfile.TemporaryDirectory() as temp_dir:
# def write_data(datasplit_type, arrays):
#     for array_key, array in arrays.items():
#         ds = prepare_ds(
#             filename=f"./tmp/{datasplit_type}.zarr",
#             ds_name=array_key.identifier,
#             total_roi=array.spec.roi,
#             voxel_size=array.spec.voxel_size,
#             dtype=array.spec.dtype,
#             delete=True
#         )
#         ds.data[:] = array.data


# # write_data("training", training_batch)
# write_data("validation", validation_batch)
# type: files
# runs_base_dir: /path/to/my/data/storage
# ```
# The `runs_base_dir` defines where your on-disk data will be stored. The `type` setting determines the database backend. The default is `files`, which stores the data in a file tree on disk. Alternatively, you can use `mongodb` to store the data in a MongoDB database. To use MongoDB, you will need to provide a `mongodbhost` and `mongodbname` in the configuration file:
#
# ```yaml
# mongodbhost: mongodb://dbuser:dbpass@dburl:dbport/
# mongodbname: dacapo

# %%
# First we need to create a config store to store our configurations
from dacapo.store.create_store import create_config_store

config_store = create_config_store()

# %%
# Then let's make sure we have data to train on
from pathlib import Path
from dacapo import Options
from examples.utils import get_viewer
from examples.synthetic_source_worker import generate_synthetic_dataset
from funlib.geometry import Coordinate
from funlib.persistence import open_ds

options = Options.instance()
runs_base_dir = options.runs_base_dir

# First for training data
train_data_path = Path(runs_base_dir, "example_train.zarr")
force = True
try:
    assert not force
    raw_array = open_ds(str(train_data_path), "raw")
    labels_array = open_ds(str(train_data_path), "labels")
except:
    train_shape = Coordinate((512, 512, 512))
    generate_synthetic_dataset(train_data_path, shape=train_shape, overwrite=True)
    raw_array = open_ds(str(train_data_path), "raw")
    labels_array = open_ds(str(train_data_path), "labels")

get_viewer(raw_array, labels_array)

# %%
# Then for validation data
validate_data_path = Path(runs_base_dir, "example_validate.zarr")
force = False
try:
    assert not force
    raw_array = ZarrArray.open_from_array_identifier(
        LocalArrayIdentifier(validate_data_path, "raw")
    )
    labels_array = ZarrArray.open_from_array_identifier(
        LocalArrayIdentifier(validate_data_path, "labels")
    )
except:
    validate_shape = Coordinate((256, 256, 256))
    generate_synthetic_dataset(validate_data_path, shape=validate_shape, overwrite=True)

get_viewer(raw_array, labels_array)

# %%
# TODO: REMOVE BELOW =============================================
from examples.random_source_pipeline import random_source_pipeline
import gunpowder as gp

pipeline, request = random_source_pipeline()


def batch_generator():
    with gp.build(pipeline):
        while True:
            yield pipeline.request_batch(request)


batch_gen = batch_generator()
batch = next(batch_gen)
raw_array = batch.arrays[gp.ArrayKey("RAW")]
labels_array = batch.arrays[gp.ArrayKey("LABELS")]

get_viewer(raw_array, labels_array)

# labels_data = labels_array.data
# raw_data = raw_array.data

# neuroglancer.set_server_bind_address("0.0.0.0")
# viewer = neuroglancer.Viewer()
# with viewer.txn() as state:
#     state.showSlices = False
#     state.layers["segs"] = neuroglancer.SegmentationLayer(
#         # segments=[str(i) for i in np.unique(data[data > 0])], # this line will cause all objects to be selected and thus all meshes to be generated...will be slow if lots of high res meshes
#         source=neuroglancer.LocalVolume(
#             data=labels_data,
#             dimensions=neuroglancer.CoordinateSpace(
#                 names=["z", "y", "x"],
#                 units=["nm", "nm", "nm"],
#                 scales=labels_array.spec.voxel_size,
#             ),
#             # voxel_offset=ds.roi.begin / ds.voxel_size,
#         ),
#         segments=np.unique(labels_data[labels_data > 0]),
#     )

#     state.layers["raw"] = neuroglancer.ImageLayer(
#         source=neuroglancer.LocalVolume(
#             data=raw_data,
#             dimensions=neuroglancer.CoordinateSpace(
#                 names=["z", "y", "x"],
#                 units=["nm", "nm", "nm"],
#                 scales=raw_array.spec.voxel_size,
#             ),
#         ),
#     )

# IFrame(src=viewer, width=1500, height=600)

# TODO: REMOVE ABOVE=============================================

# %% [markdown]
# ## Datasplit
# Where can you find your data? What format is it in? Does it need to be normalized? What data do you want to use for validation?

# We'll assume your data is in a zarr file, and that you have a raw and a ground truth dataset, all stored in your `runs_base_dir` as `example_{type}.zarr` where `{type}` is either `train` or `validate`.
# NOTE: You may need to delete old config stores if you are re-running this cell with modifications to the configs. The config names are unique and will throw an error if you try to store a config with the same name as an existing config. For the `files` backend, you can delete the `runs_base_dir/configs` directory to remove all stored configs.

# %%
from dacapo.experiments.datasplits.datasets.arrays import (
    BinarizeArrayConfig,
    ZarrArrayConfig,
)
from dacapo.experiments.datasplits import TrainValidateDataSplitConfig
from dacapo.experiments.datasplits.datasets import RawGTDatasetConfig
from pathlib import PosixPath

datasplit_config = TrainValidateDataSplitConfig(
    name="example_synthetic_datasplit_config",
    train_configs=[
        RawGTDatasetConfig(
            name="training_config",
            raw_config=ZarrArrayConfig(
                name="raw",
                file_name=PosixPath("./tmp/training.zarr"),
                dataset="RAW",
                snap_to_grid=(8, 8, 8),
                axes=None,
            ),
            gt_config=BinarizeArrayConfig(
                name="training_gt",
                source_array_config=ZarrArrayConfig(
                    name="gt",
                    file_name=PosixPath("./tmp/training.zarr"),
                    dataset="LABELS",
                    snap_to_grid=(8, 8, 8),
                    axes=None,
                ),
                groupings=[("labels", [])],
                background=0,
            ),
        ),
    ],
    validate_configs=[
        RawGTDatasetConfig(
            name="validation_config",
            raw_config=ZarrArrayConfig(
                name="raw",
                file_name=PosixPath("./tmp/validation.zarr"),
                dataset="RAW",
                snap_to_grid=(8, 8, 8),
                axes=None,
            ),
            gt_config=BinarizeArrayConfig(
                name="validation_gt",
                source_array_config=ZarrArrayConfig(
                    name="gt",
                    file_name=PosixPath("./tmp/validation.zarr"),
                    dataset="LABELS",
                    snap_to_grid=(8, 8, 8),
                    axes=None,
                ),
                groupings=[("labels", [])],
                background=0,
            ),
        )
    ],
)

config_store.store_datasplit_config(datasplit_config)

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
config_store.store_task_config(task_config)

# %% [markdown]
# ## Architecture
#
# The setup of the network you will train. Biomedical image to image translation often utilizes a UNet, but even after choosing a UNet you still need to provide some additional parameters. How much do you want to downsample? How many convolutional layers do you want?

# %%
from dacapo.experiments.architectures import CNNectomeUNetConfig

architecture_config = CNNectomeUNetConfig(
    name="example_synthetic_unet",
    input_shape=(216, 216, 216),
    fmaps_out=24,
    fmaps_in=1,
    num_fmaps=12,
    fmap_inc_factor=2,
    downsample_factors=[(2, 2, 2), (3, 3, 3), (3, 3, 3)],
    eval_shape_increase=(72, 72, 72),
)
config_store.store_architecture_config(architecture_config)

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
    name="default",
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
    clip_raw=True,
)
config_store.store_trainer_config(trainer_config)

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

iterations = 200
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
    config_store.store_run_config(run_config)

# %% [markdown]
# ## Train

# To train one of the runs, you can either do it by first creating a **Run** directly from the run config
# NOTE: The run stats are stored in the `runs_base_dir/stats` directory. You can delete this directory to remove all stored stats if you want to re-run training. Otherwise, the stats will be appended to the existing files, and the run won't start from scratch. This may cause errors
# %%
from dacapo.train import train_run

run = Run(config_store.retrieve_run_config(run_config.name))
train_run(run)

# %% [markdown]
# If you want to start your run on some compute cluster, you might want to use the command line interface: dacapo train -r {run_config.name}. This makes it particularly convenient to run on compute nodes where you can specify specific compute requirements.
