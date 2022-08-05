.. _sec_tutorial_simple_experiment_dashboard:

.. automodule:: dacapo

.. contents::
  :depth: 1
  :local:

Tutorial: A Simple Experiment using the Dashboard
-------------------------------------------------

This tutorial goes through all the necessary steps from installation
to getting an experiment running with dacapo. As an example we will learn
neuron segmentation on the `cremi dataset <https://cremi.org/data/>` using
a *3D U-Net*.

Installation
^^^^^^^^^^^^

First, follow the installation
guide:

`pip install dacapo`

Also install the dashboard:

`pip install dacapo-dashboard`

Data Storage
^^^^^^^^^^^^

Next you much choose where you want to store your data. We have 2 supported
modes of saving data.

1. Save to disk: For particularly large data such as model weights or image
volumes, it doesn't make sense to store your data in the cloud. In These
cases we store to disk.
2. Save to MongoDB: For dense data that can be nicely indexed, we encourage
saving to a mongodb. This includes data such as loss and validation scores
during training. This will allow us to quickly fetch specific scores for a
range of experiments for comparison. Note: this option requires some set up,
you need a mongodb accessible and you need to configure `DaCapo` to know
where to find it. If you just want to get started quickly, you can save
all data to disk.

`DaCapo` has a couple main data storage components:

1. Loss stats: We store the loss per iteration, and include a couple other
statistics such as how long that iteration took to compute. These will
be stored in the MongoDB if available.

2. Validation scores: For each `:ref:Run<sec_api_Run>` we will evaluate on every held out
validation dataset every `n` iterations where `n` is defined as the
validation interval on the `:ref:RunConfig<sec_api_RunConfig>`. These
will be stored in the MongoDB if available.

3. Validation volumes: For qualitative inspection, we also store the results
of validation in zarr datasets. This allows you to view the best predictions on
your held out data according to the validation metric of your choice.
This data will be stored on disk.

4. Checkpoints: These are copies of your model at various intervals during
training. Storing checkpoints lets you retrieve the best performing model
according to the validation metric of your choice.
This data will be stored on disk.

5. Training Snapshots: Every `n` iterations where `n` corresponds to the
`snapshot_interval` defined in the `:ref:TrainerConfig<sec_api_TrainerConfig>`,
we store a snapshot that includes the inputs and outputs of the model at that
iteration, and also some extra results that can be very helpful for debugging.
Saved arrays include: Ground Truth, Target (Ground Truth transformed by Task),
Raw, Prediction, Gradient, and Weights (for modifying the loss)
This data will be stored on disk.

6. Configs: To make our runs easily reproducible, we save our configuration
files and then use them to execute our experiments. This way other people
can use the exact same configuration files or change single parameters and
get comparable results. This data will be stored in the MongoDB if available.

To define where this data goes, create a `dacapo.yaml` configuration file.
Here is a template:

.. code-block:: yaml
  :caption: dacapo.yaml template

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
In this tutorial we will use the dacapo dashboard to do this.

Start the dashboard from the command line:

`dacapo-dashboard dashboard`
