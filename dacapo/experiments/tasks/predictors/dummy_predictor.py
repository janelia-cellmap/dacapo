from .predictor import Predictor


class DummyPredictor(Predictor):

    def __init__(self, embedding_dims):
        self.embedding_dims = embedding_dims

    def create_model(self, architecture, dataset):
        pass

    def create_target(self, gt):
        pass
