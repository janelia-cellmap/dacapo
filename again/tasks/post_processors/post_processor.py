import time


class PostProcessor:

    def __init__(self, parameter_range):
        self.parameter_range = parameter_range

    def set_prediction(self, prediction):
        '''To be implemented in subclasses. This function will be called before
        repeated calls to ``process`` and allows the post-processor to carry
        out general post-processing that does not depend on parameters.'''
        pass

    def process(self, prediction, parameters):
        raise Exception("To be implemented in subclasses")

    def enumerate(self, prediction):
        '''Enumerate all parameter combinations and process the predictions.
        Yields tuples of ``(parameter, post_processed)``.'''

        self.set_prediction(prediction)

        for parameters in self.parameter_range:
            if not self.reject_parameters(parameters):
                print(f"Post-processing prediction with {parameters}...")
                start = time.time()
                post_processed = self.process(
                    prediction,
                    parameters)
                print(f"...done ({time.time() - start}s)")
                yield parameters, post_processed

    def reject_parameters(self, parameters):
        '''To be implemented in subclasses to reject parameter configurations
        that should be skipped.'''
        return False
