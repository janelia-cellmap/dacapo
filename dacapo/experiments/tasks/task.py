class Task:

    def __init__(self, predictor, loss, post_processor):

        self.predictor = predictor
        self.loss = loss
        self.post_processor = post_processor
