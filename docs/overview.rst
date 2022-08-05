.. _sec_overview:

Overview
========

What is DaCapo?
^^^^^^^^^^^^^^^

DaCapo is a framework that allows for easy configuration and
execution of established machine learning techniques on
arbitrarily large volumes of multi-dimensional images.

DaCapo has 4 major configurable components:

1. :ref: `DataSplit <sec_api_datasplit>`

2. :ref: `Architecture <sec_api_architecture>`

3. :ref: `Task <sec_api_task>`

4. :ref: `Trainer <sec_api_trainer>`

These are then combined in a single :ref:`Run <sec_api_run>` that
includes your starting point (whether you want to start training from
scratch or continue off of a previously trained model) and stopping
criterion (the number of iterations you want to train).

How does DaCapo work?
^^^^^^^^^^^^^^^^^^^^^

Many machine learning experiments can be broken down into a few major components:

  1. **DataSplit**: Where can you find your data? What format is it in? Does it need
  to be normalized? What data do you want to use for validation?
  
  2. **Architecture**: Do you want to use a :ref:`UNet<spec_api_architectures_unet>`?
  How much do you want to downsample? How many convolutional layers do you want?
  All of these decisions are encoded into the
  :ref:`ArchitectureConfig<spec_api_architectureconfig>`.

  3. **Task**: What do you want to learn? An instance segmentation? If so how? Affinities,
  Distance Transform, Foreground/Background, etc. Each of these tasks are commonly learned
  and evaluated with specific loss functions and evaluation metrics. Some tasks may
  also require specific non-linearities or output formats from your model. DaCapo will
  handle all of that using the :ref:`TaskConfig<sec_api_taskconfig>` you define.
  
  4. **Trainer**: How do you want to train? This config defines the training loop
  and how the other three components work together. What sort of augmentations
  to apply during training, what learning rate and optimizer to use, what batch size
  to train with.

DaCapo allows you to define each of these configurations separately, and give them
unique names. These configurations are then stored in a mongodb, allowing you to
retrieve configs by name and easily start multitudes of jobs as combinations of
datasplits, architectures, tasks, and trainers.

The :ref:`Simple Experiment using Python<sec_tutorial_simple_experiment_python>` demonstrates how such
an experiment is assembled in ``dacapo``
