from abc import ABC
import logging

logger = logging.getLogger(__file__)


class Start(ABC):
    def __init__(self, start_config):
        self.run = start_config.run
        self.criterion = start_config.criterion

    def initialize_weights(self, model):
        from dacapo.store.create_store import create_weights_store

        weights_store = create_weights_store()
        weights = weights_store._retrieve_weights(self.run, self.criterion)
        logger.info(f"loading weights from run {self.run}, criterion: {self.criterion}")
        # load the model weights (taken from torch load_state_dict source)
        try:
            model.load_state_dict(weights.model)
        except RuntimeError as e:
            logger.warning(e)
            # if the model is not the same, we can try to load the weights
            # of the common layers
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in weights.model.items() if k in model_dict and v.size() == model_dict[k].size()}
            model_dict.update(pretrained_dict)  # update only the existing and matching layers
            model.load_state_dict(model_dict)
            logger.warning(f"loaded only common layers from weights")
