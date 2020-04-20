class PostProcessor:

    def __init__(self, parameter_range):
        self.parameter_range = parameter_range

    def process(self, prediction, parameters, store_results=None):
        raise Exception("To be implemented in subclasses")

    def enumerate(self, prediction, store_results=None):
        '''Enumerate all parameter combinations and process the predictions.
        Yields tuples of ``(parameter, post_processed)``.'''

        for parameters in self.parameter_range:
            if not self.reject_parameters(parameters):
                post_processed = self.process(
                    prediction,
                    parameters,
                    store_results)
                yield parameters, post_processed

    def reject_parameters(self, parameters):
        '''To be implemented in subclasses to reject parameter configurations
        that should be skipped.'''
        return False
