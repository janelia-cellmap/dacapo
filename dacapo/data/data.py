class Data:
    def __init__(self, data_config):

        print(f"DATA CONFIG: {data_config.to_dict()}")
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
