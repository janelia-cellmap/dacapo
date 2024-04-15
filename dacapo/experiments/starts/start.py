from abc import ABC
import logging

logger = logging.getLogger(__file__)

head_keys = [
    "prediction_head.weight",
    "prediction_head.bias",
    "chain.1.weight",
    "chain.1.bias",
]


def match_heads(model, head_weights, old_head, new_head):
    """
    Matches the head of the model to the new head by copying the weights
    of the old head to the new head. The weights of the old head are
    copied to the new head by matching the labels of the old head to the
    labels of the new head.

    Args:
        model : obj
            The model to which the weights are to be loaded.
        head_weights : dict
            The weights of the old head.
        old_head : list
            The labels of the old head.
        new_head : list
            The labels of the new head.
    Returns:
        model : obj
            The model with the weights of the old head copied to the new
            head.
    Raises:
        RuntimeError
            If the old head is not found in the new head, a RuntimeError
            exception is thrown which is logged and handled by loading
            only the common layers from weights.
    Examples:
        >>> model = match_heads(model, head_weights, old_head, new_head)
    Notes:
        This function is called by the Start class to match the head of
        the model to the new head by copying the weights of the old head
        to the new head.
    """
    for label in new_head:
        if label in old_head:
            logger.warning(f"matching head for {label}.")
            old_index = old_head.index(label)
            new_index = new_head.index(label)
            for key in head_keys:
                if key in model.state_dict().keys():
                    new_value = head_weights[key][old_index]
                    model.state_dict()[key][new_index] = new_value
            logger.warning(f"matched head for {label}.")


def _set_weights(model, weights, run, criterion, old_head=None, new_head=None):
    """
    Loads the weights of the model from the dacapo store into the model. If
    the old head and new head are provided, the weights of the old head are
    copied to the new head by matching the labels of the old head to the labels
    of the new head. If the old head is not found in the new head, a RuntimeError
    exception is thrown which is logged and handled by loading only the common
    layers from weights.

    Args:
        model : obj
            The model to which the weights are to be loaded.
        weights : obj
            The weights of the model retrieved from the dacapo store.
        run : str
            The specified run to retrieve weights for the model.
        criterion : str
            The policy that was used to decide when to store the weights.
        old_head : list
            The labels of the old head.
        new_head : list
            The labels of the new head.
    Returns:
        model : obj
            The model with the weights loaded from the dacapo store.
    Raises:
        RuntimeError
            If weights of a non-existing or mismatched layer are being
            loaded, a RuntimeError exception is thrown which is logged
            and handled by loading only the common layers from weights.
    Examples:
        >>> model = _set_weights(model, weights, run, criterion, old_head, new_head)
    Notes:
        This function is called by the Start class to load the weights of the
        model from the dacapo store into the model. If the old head and new head
        are provided, the weights of the old head are copied to the new head by
        matching the labels of the old head to the labels of the new head. If the
        old head is not found in the new head, a RuntimeError exception is thrown
        which is logged and handled by loading only the common layers from weights.
    """
    logger.warning(
        f"loading weights from run {run}, criterion: {criterion}, old_head {old_head}, new_head: {new_head}"
    )
    try:
        if old_head and new_head:
            try:
                logger.warning(f"matching heads from run {run}, criterion: {criterion}")
                logger.warning(f"old head: {old_head}")
                logger.warning(f"new head: {new_head}")
                head_weights = {}
                for key in head_keys:
                    head_weights[key] = weights.model[key]
                for key in head_keys:
                    weights.model.pop(key, None)
                try:
                    model.load_state_dict(weights.model, strict=True)
                except:
                    logger.warning(
                        "Unable to load model in strict mode. Loading flexibly."
                    )
                    model.load_state_dict(weights.model, strict=False)
                model = match_heads(model, head_weights, old_head, new_head)
            except RuntimeError as e:
                logger.error(f"ERROR starter matching head: {e}")
                logger.warning(f"removing head from run {run}, criterion: {criterion}")
                for key in head_keys:
                    weights.model.pop(key, None)
                model.load_state_dict(weights.model, strict=False)
                logger.warning(
                    f"loaded weights in non strict mode from run {run}, criterion: {criterion}"
                )
        else:
            try:
                model.load_state_dict(weights.model)
            except RuntimeError as e:
                logger.warning(e)
                model_dict = model.state_dict()
                pretrained_dict = {
                    k: v
                    for k, v in weights.model.items()
                    if k in model_dict and v.size() == model_dict[k].size()
                }
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                logger.warning(f"loaded only common layers from weights")
    except RuntimeError as e:
        logger.warning(f"ERROR starter: {e}")


class Start(ABC):
    """
    This class interfaces with the dacapo store to retrieve and load the
    weights of the starter model used for finetuning.

    Attributes:
        run : str
            The specified run to retrieve weights for the model.
        criterion : str
            The policy that was used to decide when to store the weights.
        channels : int
            The number of channels in the input data.
    Methods:
        __init__(start_config)
            Initializes the Start class with specified config to run the
            initialization of weights for a model associated with a specific
            criterion.
        initialize_weights(model, new_head=None)
            Retrieves the weights from the dacapo store and load them into
            the model.
    Notes:
        This class is used to retrieve and load the weights of the starter
        model used for finetuning from the dacapo store.
    """

    def __init__(self, start_config):
        """
        Initializes the Start class with specified config to run the
        initialization of weights for a model associated with a specific
        criterion.

        Args:
            start_config : obj
                The configuration to initialize the Start class.
        Examples:
            >>> start = Start(start_config)
        Notes:
            This function is called by the Start class to initialize the
            Start class with specified config to run the initialization of
            weights for a model associated with a specific criterion.
        """
        # Old version return a dict, new version return an object, this line is to support both
        if isinstance(start_config, dict):
            self.run = start_config["run"]
            self.criterion = start_config["criterion"]
        else:
            self.run = start_config.run
            self.criterion = start_config.criterion

        self.channels = None

        if hasattr(start_config, "task_config"):
            if hasattr(start_config.task_config, "channels"):
                self.channels = start_config.task_config.channels

    def initialize_weights(self, model, new_head=None):
        """
        Retrieves the weights from the dacapo store and load them into
        the model.

        Args:
            model : obj
                The model to which the weights are to be loaded.
            new_head : list
                The labels of the new head.
        Returns:
            model : obj
                The model with the weights loaded from the dacapo store.
        Raises:
            RuntimeError
                If weights of a non-existing or mismatched layer are being
                loaded, a RuntimeError exception is thrown which is logged
                and handled by loading only the common layers from weights.
        Examples:
            >>> model = start.initialize_weights(model, new_head)
        Notes:
            This function is called by the Start class to retrieve the weights
            from the dacapo store and load them into the model.
        """
        from dacapo.store.create_store import create_weights_store

        weights_store = create_weights_store()
        weights = weights_store._retrieve_weights(self.run, self.criterion)
        _set_weights(model, weights, self.run, self.criterion, self.channels, new_head)
