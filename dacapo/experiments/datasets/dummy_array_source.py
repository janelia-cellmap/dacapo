from .array_source import ArraySource


class DummyArraySource(ArraySource):
    """This is just a dummy array source for testing."""

    def __init__(self, source_config):

        self.filename = source_config.filename

    def axes(self):
        return 'czyx'

    def dims(self):
        return 3
