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

# %% [markdown]
# ## Datasplit
# Where can you find your data? What format is it in? Does it need to be normalized? What data do you want to use for validation?
# %%
from dacapo.store.create_store import create_config_store

config_store = create_config_store()
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
    name="example_synthetic_distance_task_config",
    channels=["labels"],
    clip_distance=80.0,
    tol_distance=80.0,
    scale_factor=160.0,
    mask_distances=True,
    clipmin=0.05,
    clipmax=0.95,
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
    name="example_synthetic_trainer_config",
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

iterations = 100000
validation_interval = 50
repetitions = 1
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

# %% [markdown]
# ## Train

# %% [markdown]
# To train one of the runs, you can either do it by first creating a **Run** directly from the run config

# %%
from dacapo.train import train_run


run_name = "example_scratch_example_synthetic_datasplit_config_example_synthetic_distance_task_config_example_synthetic_unet_example_synthetic_trainer_config__0"

run = Run(config_store.retrieve_run_config(run_name))
train_run(run)

# %% [markdown]
# If you want to start your run on some compute cluster, you might want to use the command line interface: dacapo train -r {run_config.name}. This makes it particularly convenient to run on compute nodes where you can specify specific compute requirements.

# %%
# from dacapo.validate import validate

# validate(
#     "example_scratch_example_synthetic_datasplit_config_example_synthetic_distance_task_config_example_synthetic_unet_example_synthetic_trainer_config__0",
#     iteration=10000,
# )

# %%
