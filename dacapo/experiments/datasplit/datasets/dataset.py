class Dataset:
    """A dataset consisting of train, validate, and predict sources."""

    def __init__(self, dataset_config):

        self.train = self.__create_sources(dataset_config.train_sources)
        self.validate = self.__create_sources(dataset_config.validate_sources)
        self.predict = self.__create_sources(dataset_config.predict_sources)

    def __create_sources(self, sources_config):

        sources = {}

        for key, source_config in sources_config.items():

            source_type = source_config.source_type
            source = source_type(source_config)
            sources[key] = source

        return sources
