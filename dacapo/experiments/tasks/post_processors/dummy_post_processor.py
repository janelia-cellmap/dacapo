from .post_processor import PostProcessor


class DummyPostProcessor(PostProcessor):

    def __init__(self, detection_threshold):
        self.detection_threshold = detection_threshold

    def process(self, container, prediction_dataset, output_dataset):
        pass
