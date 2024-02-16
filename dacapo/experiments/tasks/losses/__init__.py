"""
dacapo losses scripts - imports various loss functions from the library.

This module consists of classes importing several loss calculation methods used in deep learning.

Functions:
:func: .dummy_loss.DummyLoss - A placeholder for a loss function, performs no real calculation.
:func: .mse_loss.MSELoss - Calculates the Mean Squared Error loss between predicted and actual values.
:func: .loss.Loss - Generic loss function base class.
:func: .affinities_loss.AffinitiesLoss - Calculates the loss due to differing input and output affinities.
:func: .hot_distance_loss.HotDistanceLoss - Calculates the loss based on the distances between hot points in the data.

Note: The 'noqa' comments are used to instruct flake8 to ignore these lines for linting purposes.

"""

from .dummy_loss import DummyLoss  # noqa
from .mse_loss import MSELoss  # noqa
from .loss import Loss  # noqa
from .affinities_loss import AffinitiesLoss  # noqa
from .hot_distance_loss import HotDistanceLoss  # noqa