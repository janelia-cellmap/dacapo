import itertools

from .post_processing_parameters import PostProcessingParameters


class PostProcessingParameterRange:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.names = kwargs.keys()
        self.ranges = kwargs.values()

    def __iter__(self):

        # no parameters
        if not self.names:
            yield PostProcessingParameters()
            return

        for id_, values in enumerate(itertools.product(*self.ranges)):
            kwargs = {
                name: value
                for name, value in zip(self.names, values)
            }
            yield PostProcessingParameters(id_, **kwargs)
