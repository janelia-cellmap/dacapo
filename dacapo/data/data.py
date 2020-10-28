import itertools

from .group import TrainValidateSplit, TestSplit, ArrayGroup, GraphGroup


class Data:
    def __init__(self, data_config):

        arrays = data_config.arrays
        graphs = data_config.graphs

        for array in arrays:
            self.__setattr__(array, TrainValidateSplit(array))
        for graph in graphs:
            self.__setattr__(graph, TrainValidateSplit(graph))

        for key in arrays:
            train_source = self.array_source(data_config.train_sources[key])
            getattr(self, key).train = train_source
            validate_source = self.array_source(data_config.validate_sources[key])
            getattr(self, key).validate = validate_source

        for key in graphs:
            train_source = self.graph_source(data_config.train_sources[key])
            getattr(self, key).train = train_source
            validate_source = self.graph_source(data_config.validate_sources[key])
            getattr(self, key).validate = validate_source

    def array_source(self, source):
        if isinstance(source, list):
            return ArrayGroup(list(source))
        else:
            return source

    def graph_source(self, source):
        if isinstance(source, list):
            return GraphGroup(list(source))
        else:
            return source


class PredictData:
    def __init__(self, data_config):

        arrays = data_config.arrays
        graphs = data_config.graphs

        for array in arrays:
            self.__setattr__(array, TestSplit(array))
        for graph in graphs:
            self.__setattr__(graph, TestSplit(graph))

        for key in arrays:
            test_source = self.array_source(data_config.test_sources[key])
            getattr(self, key).test = test_source

        for key in graphs:
            test_source = self.graph_source(data_config.test_sources[key])
            getattr(self, key).train = test_source

    def array_source(self, source):
        if isinstance(source, list):
            return ArrayGroup(list(source))
        else:
            return source

    def graph_source(self, source):
        if isinstance(source, list):
            return GraphGroup(list(source))
        else:
            return source
