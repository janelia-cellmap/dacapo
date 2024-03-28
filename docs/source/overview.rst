.. _sec_overview:

Overview
========

What is DaCapo?
^^^^^^^^^^^^^^^

DaCapo is a framework that allows for easy configuration and
execution of established machine learning techniques on
arbitrarily large volumes of multi-dimensional images.

DaCapo has 4 major configurable components:

1. :class:`dacapo.datasplits.DataSplit`

2. :class:`dacapo.architectures.Architecture`

3. :class:`dacapo.tasks.Task`

4. :class:`dacapo.trainers.Trainer`

These are then combined in a single :class:`dacapo.experiments.Run` that
includes your starting point (whether you want to start training from
scratch or continue off of a previously trained model) and stopping
criterion (the number of iterations you want to train).

How does DaCapo work?
^^^^^^^^^^^^^^^^^^^^^

Each of the major components can be configured separately allowing you to define
your job in a nicely structured format. Here we define what each component is
responsible for:

1. | **DataSplit**: Where can you find your data? What format is it in? Does it need
   | to be normalized? What data do you want to use for validation?

2. | **Architecture**: Biomedical image to image translation often utilizes a UNet,
   | but even after choosing a UNet you still need to provide some additional parameters.
   | How much do you want to downsample? How many convolutional layers do you want?

3. | **Task**: What do you want to learn? An instance segmentation? If so how? Affinities,
   | Distance Transform, Foreground/Background, etc. Each of these tasks are commonly learned
   | and evaluated with specific loss functions and evaluation metrics. Some tasks may
   | also require specific non-linearities or output formats from your model.

4. | **Trainer**: How do you want to train? This config defines the training loop
   | and how the other three components work together. What sort of augmentations
   | to apply during training, what learning rate and optimizer to use, what batch size
   | to train with.

DaCapo allows you to define each of these configurations separately, and give them
unique names. These configurations are then stored in a mongodb or on your filesystem,
allowing you to retrieve configs by name and easily start multitudes of jobs as
combinations of `Datasplits`, `Architectures`, `Tasks`, and `Trainers`.

The :ref:`Simple Experiment using Python<sec_tutorial_simple_experiment_python>` demonstrates how such
an experiment is assembled in ``dacapo``