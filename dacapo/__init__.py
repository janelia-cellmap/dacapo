"""
dacapo module
==============

This module contains several useful methods for performing common tasks in dacapo library.

Modules:
-----------
Options            - Deals with configuring aspects of the program's operations.
experiments        - This module is responsible for conducting experiments.
apply              - Applies the results of the training process to the given dataset.
train              - Trains the model using given data set.
validate           - This module is for validating the model.
predict            - This module is used to generate predictions based on the model.

"""

from .options import Options  # noqa
from . import experiments  # noqa
from .apply import apply  # noqa
from .train import train  # noqa
from .validate import validate  # noqa
from .predict import predict  # noqa

