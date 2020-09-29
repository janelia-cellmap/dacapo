import itertools

from .group import TrainValidateSplit, ArrayGroup, GraphGroup


class Data:
    def __init__(self, data_config):

        print(f"DATA CONFIG: {str(data_config)}")
        arrays = data_config.arrays
        graphs = data_config.graphs

        for array in arrays:
            self.__setattr__(array, TrainValidateSplit(array))
        for graph in graphs:
            self.__setattr__(graph, TrainValidateSplit(graph))

        for key in arrays:
            train_source = ArrayGroup(list(data_config.train_sources[key]))
            getattr(self, key).train = train_source
            validate_source = ArrayGroup(list(data_config.validate_sources[key]))
            getattr(self, key).validate = validate_source

        for key in graphs:
            train_source = GraphGroup(list(data_config.train_sources[key]))
            getattr(self, key).train = train_source
            validate_source = GraphGroup(list(data_config.validate_sources[key]))
            getattr(self, key).validate = validate_source
