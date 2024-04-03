.. _sec_tutorial_simple_experiment_python:

.. automodule:: dacapo

.. contents::
  :depth: 1
  :local:

Tutorial: A Simple Experiment in Python
---------------------------------------

This tutorial goes through all the necessary steps from installation
to getting an experiment running with dacapo. As an example we will learn
neuron segmentation on the `cremi dataset <https://cremi.org/data/>` using
a *3D U-Net*.

Installation
^^^^^^^^^^^^

First, follow the :ref:`installation guide<sec_install>`.

Data Storage
^^^^^^^^^^^^

Next you much choose where you want to store your data. We have 2 supported
modes of saving data.

1. | Save to disk: For particularly large data such as model weights or image
   | volumes, it doesn't make sense to store your data in the cloud. In These
   | cases we store to disk.
2. | Save to MongoDB: For dense data that can be nicely indexed, we encourage
   | saving to a mongodb. This includes data such as loss and validation scores
   | during training. This will allow us to quickly fetch specific scores for a
   | range of experiments for comparison. Note: this option requires some set up,
   | you need a mongodb accessible and you need to configure `DaCapo` to know
   | where to find it. If you just want to get started quickly, you can save
   | all data to disk.

`DaCapo` has a couple main data storage components:

1. | Loss stats: We store the loss per iteration, and include a couple other
   | statistics such as how long that iteration took to compute. These will
   | be stored in the MongoDB if available.

2. | Validation scores: For each `:ref:Run<sec_api_Run>` we will evaluate on every held out
   | validation dataset every `n` iterations where `n` is defined as the
   | validation interval on the `:ref:RunConfig<sec_api_RunConfig>`. These
   | will be stored in the MongoDB if available.

3. | Validation volumes: For qualitative inspection, we also store the results
   | of validation in zarr datasets. This allows you to view the best predictions on
   | your held out data according to the validation metric of your choice.
   | This data will be stored on disk.

4. | Checkpoints: These are copies of your model at various intervals during
   | training. Storing checkpoints lets you retrieve the best performing model
   | according to the validation metric of your choice.
   | This data will be stored on disk.

5. | Training Snapshots: Every `n` iterations where `n` corresponds to the
   | `snapshot_interval` defined in the `:ref:TrainerConfig<sec_api_TrainerConfig>`,
   | we store a snapshot that includes the inputs and outputs of the model at that
   | iteration, and also some extra results that can be very helpful for debugging.
   | Saved arrays include: Ground Truth, Target (Ground Truth transformed by Task),
   | Raw, Prediction, Gradient, and Weights (for modifying the loss)
   | This data will be stored on disk.

6. | Configs: To make our runs easily reproducible, we save our configuration
   | files and then use them to execute our experiments. This way other people
   | can use the exact same configuration files or change single parameters and
   | get comparable results. This data will be stored in the MongoDB if available.

To define where this data goes, create a `dacapo.yaml` configuration file.
Here is a template:

.. code-block:: yaml

    mongodbhost: mongodb://dbuser:dbpass@dburl:dbport/
    mongodbname: dacapo
    runs_base_dir: /path/to/my/data/storage

The `runs_base_dir` defines where your on-disk data will be stored.
The `mongodbhost` and `mongodbname` define the mongodb host and database
that will store your cloud data.
If you want to store everything on disk, replace `mongodbhost` and `mongodbname`
with a single `type: files` and everything will be saved to disk.

Configs
^^^^^^^

Next you need to create your configuration files for your experiments.
This can all be done in python. There is also a web based gui: the
`dacapo-dashboard <https://github.com/funkelab/dacapo-dashboard>`, see the
:ref:`Simple Experiment using the Dashboard<sec_tutorial_simple_experiment_dashboard>`
for a tutorial stepping through the use of the dashboard.

First lets handle some basics, importing, setting logging, etc.

.. code-block:: python

    import dacapo
    import logging

    logging.basicConfig(level=logging.INFO)

Now lets create some configs.

#.  DataSplit
    
    .. code-block:: python
   
      # TODO: create datasplit config
      datasplit_config = ...

#.  Architecture

    .. code-block:: python

      from dacapo.experiments.architectures import CNNectomeUNetConfig
      architecture_config = CNNectomeUNetConfig(
          name="small_unet",
          input_shape=Coordinate(212, 212, 212),
          eval_shape_increase=Coordinate(72, 72, 72),
          fmaps_in=1,
          num_fmaps=8,
          fmaps_out=32,
          fmap_inc_factor=4,
          downsample_factors=[(2, 2, 2), (2, 2, 2), (2, 2, 2)],
          constant_upsample=True,
      )

#.  Task

    .. code-block:: python

      from dacapo.experiments.tasks import AffinitiesTaskConfig

      task_config = AffinitiesTaskConfig(
          name="AffinitiesPrediction",
          neighborhood=[(0,0,1),(0,1,0),(1,0,0)]
      )

#.  Trainer

    .. code-block:: python

        from dacapo.experiments.trainers import GunpowderTrainerConfig
        from dacapo.experiments.trainers.gp_augments import (
            SimpleAugmentConfig,
            ElasticAugmentConfig,
            IntensityAugmentConfig,
        )

        trainer_config = GunpowderTrainerConfig(
            name="gunpowder",
            batch_size=2,
            learning_rate=0.0001,
            augments=[
                SimpleAugmentConfig(),
                ElasticAugmentConfig(
                    control_point_spacing=(100, 100, 100),
                    control_point_displacement_sigma=(10.0, 10.0, 10.0),
                    rotation_interval=(0, math.pi / 2.0),
                    subsample=8,
                    uniform_3d_rotation=True,
                ),
                IntensityAugmentConfig(
                    scale=(0.25, 1.75),
                    shift=(-0.5, 0.35),
                    clip=False,
                )
            ],
            num_data_fetchers=20,
            snapshot_interval=10000,
            min_masked=0.15,
            min_labelled=0.1,
        )

Create a Run
^^^^^^^^^^^^

Now that we have our components configured, we just need to
combine them into a run and start training.

.. code-block:: python
  :caption: Create a run

    from funlib.geometry import Coordinate
    from dacapo.experiments.run_config import RunConfig
    from dacapo.experiments.run import Run

    from torchsummary import summary


    run_config = RunConfig(
        name="tutorial_run",
        task_config=task_config,
        architecture_config=architecture_config,
        trainer_config=trainer_config,
        datasplit_config=datasplit_config,
        repetition=0,
        num_iterations=100000,
        validation_interval=1000,
    )

    run = Run(run_config)

    # if you want a summary of the model you can print that here
    print(summary(run.model, (1, 212, 212, 212)))

Start the Run
^^^^^^^^^^^^^
You have 2 options for starting the run:

#.  Simple case:

    .. code-block:: python

        from dacapo.train import train_run

        train_run(run)

    Your job will run but you haven't stored your configuration files, so
    it may be difficult to reproduce your work without copy pasting everything
    you've written so far.

#.  Store your configs

    .. code-block:: python

        from dacapo.store.create_store import create_config_store
        from dacapo.train import train

        config_store = create_config_store()

        config_store.store_datasplit_config(datasplit_config)
        config_store.store_architecture_config(architecture_config)
        config_store.store_task_config(task_config)
        config_store.store_trainer_config(trainer_config)
        config_store.store_run_config(run_config)

        # Optional start training by config name:
        train(run_config.name)

    Once you have stored all your configs, you can start your Run just with
    the name of the config you want to start. If you want to start your
    run on some compute cluster, you might want to use the command line
    interface: :code:`dacapo train -r {run_config.name}`.
    This makes it particularly convenient to run on compute nodes where
    you can specify specific compute requirements.


Finally your job should be running. The full script to run this tutorial is
provided here:

`dacapo.yaml`:

.. code-block:: yaml

    mongodbhost: mongodb://dbuser:dbpass@dburl:dbport/
    mongodbname: dacapo
    runs_base_dir: /path/to/my/data/storage


`train.py`:

.. code-block:: python

    import dacapo
    import logging
    from dacapo.experiments.architectures import CNNectomeUNetConfig
    from dacapo.experiments.tasks import AffinitiesTaskConfig
    from dacapo.experiments.trainers import GunpowderTrainerConfig
    from dacapo.experiments.trainers.gp_augments import (
        SimpleAugmentConfig,
        ElasticAugmentConfig,
        IntensityAugmentConfig,
    )
    from funlib.geometry import Coordinate
    from dacapo.experiments.run_config import RunConfig
    from dacapo.experiments.run import Run
    from dacapo.store.create_store import create_config_store
    from dacapo.train import train

    logging.basicConfig(level=logging.INFO)
   
    # TODO: create datasplit config
    train_array
    datasplit_config = ...


    # Create Architecture Config
    architecture_config = CNNectomeUNetConfig(
        name="small_unet",
        input_shape=Coordinate(212, 212, 212),
        eval_shape_increase=Coordinate(72, 72, 72),
        fmaps_in=1,
        num_fmaps=8,
        fmaps_out=32,
        fmap_inc_factor=4,
        downsample_factors=[(2, 2, 2), (2, 2, 2), (2, 2, 2)],
        constant_upsample=True,
    )

    # Create Task Config

    task_config = AffinitiesTaskConfig(
        name="AffinitiesPrediction",
        neighborhood=[(0,0,1),(0,1,0),(1,0,0)]
    )


    # Create Trainer Config

    trainer_config = GunpowderTrainerConfig(
        name="gunpowder",
        batch_size=2,
        learning_rate=0.0001,
        augments=[
            SimpleAugmentConfig(),
            ElasticAugmentConfig(
                control_point_spacing=(100, 100, 100),
                control_point_displacement_sigma=(10.0, 10.0, 10.0),
                rotation_interval=(0, math.pi / 2.0),
                subsample=8,
                uniform_3d_rotation=True,
            ),
            IntensityAugmentConfig(
                scale=(0.25, 1.75),
                shift=(-0.5, 0.35),
                clip=False,
            )
        ],
        num_data_fetchers=20,
        snapshot_interval=10000,
        min_masked=0.15,
        min_labelled=0.1,
    )

    # Create Run Config

    run_config = RunConfig(
        name="tutorial_run",
        task_config=task_config,
        architecture_config=architecture_config,
        trainer_config=trainer_config,
        datasplit_config=datasplit_config,
        repetition=0,
        num_iterations=100000,
        validation_interval=1000,
    )

    run = Run(run_config)

    # Store configs

    config_store = create_config_store()

    config_store.store_datasplit_config(datasplit_config)
    config_store.store_architecture_config(architecture_config)
    config_store.store_task_config(task_config)
    config_store.store_trainer_config(trainer_config)
    config_store.store_run_config(run_config)

    # Optional start training by config name:
    train(run_config.name)