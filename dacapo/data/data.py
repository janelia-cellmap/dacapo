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

        for key in arrays:
            train_source = data_config.train_sources[key]
            getattr(self, key).train = train_source
            validate_source = data_config.validate_sources[key]
            getattr(self, key).validate = validate_source
            
            
        """
        for key in sorted(
            data_config.to_dict().keys(), key=lambda x: len(x.split("."))
        ):
            if key != "id":
                config = data_config.__getattr__(key)
                kwargs = config.to_dict(default_only=True)
                del kwargs["id"]
                del kwargs["dataset"]
                if len(key.split(".")) > 1:
                    components = key.split(".")
                    path = components[0:-1]
                    key = components[-1]
                    obj = self
                    for p in path:
                        obj = getattr(self, p)

                    dataset = config.dataset(**kwargs)
                    obj.__setattr__(key, dataset)
                    obj.add_dataset(dataset)

                else:
                    self.__setattr__(key, config.dataset(**kwargs))
        """
