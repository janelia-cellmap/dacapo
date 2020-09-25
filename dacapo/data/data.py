import itertools

from .group import DatasetGroup


class Data:
    def __init__(self, data_config):

        print(f"DATA CONFIG: {str(data_config)}")
        arrays = data_config.arrays
        graphs = data_config.graphs

        for array in arrays:
            self.__setattr__(array, DatasetGroup())
        for graph in graphs:
            self.__setattr__(graph, DatasetGroup())

        for key in itertools.chain(arrays, graphs):
            train_source = data_config.train_sources[key]
            getattr(self, key).train = train_source
            validate_source = data_config.validate_sources[key]
            getattr(self, key).validate = validate_source
