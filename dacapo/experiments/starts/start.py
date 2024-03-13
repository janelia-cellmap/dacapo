from abc import ABC
import logging

logger = logging.getLogger(__file__)


class Start(ABC):
    """
    This class interfaces with the dacapo store to retrieve and load the 
    weights of the starter model used for finetuning.

    Attributes
    ----------
    run : str
        The specified run to retrieve weights for the model.
    criterion : str
        The policy that was used to decide when to store the weights.
    """

    def __init__(self, start_config):
        """
        Initializes the Start class with specified config to run the 
        initialization of weights for a model associated with a specific 
        criterion.

        Parameters
        ----------
        start_config : obj
            An object containing configuration details for the model 
            initialization.
        """
        self.run = start_config.run
        self.criterion = start_config.criterion

    def initialize_weights(self, model):
        """
        Retrieves the weights from the dacapo store and load them into 
        the model.

        Parameters
        ----------
        model : obj 
            The model to which the weights are to be loaded.

        Raises
        ------
        RuntimeError
            If weights of a non-existing or mismatched layer are being 
            loaded, a RuntimeError exception is thrown which is logged 
            and handled by loading only the common layers from weights.
        """
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
            pretrained_dict = {
                k: v
                for k, v in weights.model.items()
                if k in model_dict and v.size() == model_dict[k].size()
            }
            model_dict.update(
                pretrained_dict
            )  # update only the existing and matching layers
            model.load_state_dict(model_dict)
            logger.warning(f"loaded only common layers from weights")
