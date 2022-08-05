.. _sec_api:

API Reference
=============

.. automodule:: dacapo
   :noindex:

.. _sec_api_run:

Run
___

  .. autoclass:: dacapo.experiments.Run
    :members:

RunConfigs
^^^^^^^^^^

  .. autoclass:: dacapo.experiments.RunConfig
    :members:


DataSplits
__________

  .. autoclass:: dacapo.experiments.datasplits.DataSplit
    :members:

TrainValidateDataSplit
^^^^^^^^^^^^^^^^^^^^^^

  .. autoclass:: dacapo.experiments.datasplits.TrainValidateDataSplitConfig
    :members:

  .. autoclass:: dacapo.experiments.datasplits.TrainValidateDataSplit
    :members:

Architectures
_____________

  .. autoclass:: dacapo.experiments.architectures.Architecture
    :members:

CNNectomeUNet
^^^^^^^^^^^^^
  .. autoclass:: dacapo.experiments.architectures.CNNectomeUNetConfig
    :members:

  .. autoclass:: dacapo.experiments.architectures.CNNectomeUNet
    :members:

Tasks
_____

  .. autoclass:: dacapo.experiments.tasks.Task
    :members:

OneHotTask
^^^^^^^^^^
  .. autoclass:: dacapo.experiments.tasks.OneHotTaskConfig
    :members:

  .. autoclass:: dacapo.experiments.tasks.OneHotTask
    :members:

AffinitiesTask
^^^^^^^^^^^^^^

  .. autoclass:: dacapo.experiments.tasks.AffinitiesTaskConfig
    :members:

  .. autoclass:: dacapo.experiments.tasks.AffinitiesTask
    :members:

DistanceTask
^^^^^^^^^^^^

  .. autoclass:: dacapo.experiments.tasks.DistanceTaskConfig
    :members:

  .. autoclass:: dacapo.experiments.tasks.DistanceTask
    :members:

Trainers
________

  .. autoclass:: dacapo.experiments.trainers.Trainer
    :members:

GunpowderTrainer
^^^^^^^^^^^^^^^^

  .. autoclass:: dacapo.experiments.trainers.GunpowderTrainerConfig
    :members:

  .. autoclass:: dacapo.experiments.trainers.GunpowderTrainer
    :members:



