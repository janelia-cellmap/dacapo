"""
This module imports different kinds of predictor classes from different modules. 

Attributes:
    DummyPredictor: This class is used to predict dummy values.
    DistancePredictor: This class computes and predicts distances.
    OneHotPredictor: This class predicts one hot encoded values.
    Predictor: This is the main Predictor class from which other classes inherit.
    AffinitiesPredictor: This class works with predicting affinities.
    InnerDistancePredictor: This class predicts inner distances.
    HotDistancePredictor: This class is used for hot distance predictions.
"""
from .dummy_predictor import DummyPredictor  # noqa
from .distance_predictor import DistancePredictor  # noqa
from .one_hot_predictor import OneHotPredictor  # noqa
from .predictor import Predictor  # noqa
from .affinities_predictor import AffinitiesPredictor  # noqa
from .inner_distance_predictor import InnerDistancePredictor  # noqa
from .hot_distance_predictor import HotDistancePredictor  # noqa
