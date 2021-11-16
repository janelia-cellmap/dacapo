from .dummy import dummy_run
from .one_hot_segmentation import one_hot_run
from .distances_run import distance_run

RUNS = [dummy_run, one_hot_run, distance_run]
