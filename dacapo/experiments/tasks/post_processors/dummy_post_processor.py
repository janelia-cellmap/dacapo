from .post_processor import PostProcessor


class DummyPostProcessor(PostProcessor):

    def __init__(self, detection_threshold):
        self.detection_threshold = detection_threshold

    def process(
            self,
            prediction_container,
            prediction_dataset,
            output_container,
            output_dataset,
            parameters):
        pass
