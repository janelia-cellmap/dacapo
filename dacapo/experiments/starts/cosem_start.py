from abc import ABC
import logging
from cellmap_models import cosem
from pathlib import Path
from .start import Start, _set_weights

logger = logging.getLogger(__file__)


def get_model_setup(run):
    """
    Loads the model setup from the dacapo store for the specified run. The
    model setup includes the classes_channels, voxel_size_input and
    voxel_size_output.

    Args:
        run : str
            The run for which the model setup is to be loaded.
    Returns:
        classes_channels : list
            The classes_channels of the model.
        voxel_size_input : list
            The voxel_size_input of the model.
        voxel_size_output : list
            The voxel_size_output of the model.
    Raises:
        Exception
            If the model setup cannot be loaded, an Exception is thrown which
            is logged and handled by training the model without head matching.
    Examples:
        >>> classes_channels, voxel_size_input, voxel_size_output = get_model_setup(run)
    Notes:
        This function is called by the CosemStart class to load the model setup
        from the dacapo store for the specified run.
    """
    try:
        model = cosem.load_model(run)
        if hasattr(model, "classes_channels"):
            classes_channels = model.classes_channels
        else:
            classes_channels = None
        if hasattr(model, "voxel_size_input"):
            voxel_size_input = model.voxel_size_input
        else:
            voxel_size_input = None
        if hasattr(model, "voxel_size_output"):
            voxel_size_output = model.voxel_size_output
        else:
            voxel_size_output = None
        return classes_channels, voxel_size_input, voxel_size_output
    except Exception as e:
        logger.error(
            f"could not load model setup: {e} - Not a big deal, model will train wiithout head matching"
        )
        return None, None, None


class CosemStart(Start):
    """
    A class to represent the starting point for tasks. This class inherits
    from the Start class and is used to load the weights of the starter model
    used for finetuning. The weights are loaded from the dacapo store for the
    specified run and criterion.

    Attributes:
        run : str
            The run to be used as a starting point for tasks.
        criterion : str
            The criterion to be used for choosing weights from run.
        name : str
            The name of the run and criterion.
        channels : list
            The classes_channels of the model.
    Methods:
        __init__(start_config)
            Initializes the CosemStart class with specified config to run the
            initialization of weights for a model associated with a specific
            criterion.
        check()
            Checks if the checkpoint for the specified run and criterion exists.
        initialize_weights(model, new_head=None)
            Retrieves the weights from the dacapo store and load them into
            the model.
    Notes:
        This class is used to represent the starting point for tasks. The weights
        of the starter model used for finetuning are loaded from the dacapo store.
    """

    def __init__(self, start_config):
        """
        Initializes the CosemStart class with specified config to run the
        initialization of weights for a model associated with a specific
        criterion.

        Args:
            start_config : obj
                The configuration to initialize the CosemStart class.
        Raises:
            Exception
                If the model setup cannot be loaded, an Exception is thrown which
                is logged and handled by training the model without head matching.
        Examples:
            >>> start = CosemStart(start_config)
        Notes:
            This function is called by the CosemStart class to initialize the
            CosemStart class with specified config to run the initialization of
            weights for a model associated with a specific criterion.
        """
        self.run = start_config.run
        self.criterion = start_config.criterion
        self.name = f"{self.run}/{self.criterion}"
        channels, voxel_size_input, voxel_size_output = get_model_setup(self.run)
        if voxel_size_input is not None:
            logger.warning(
                f"Starter model resolution: input {voxel_size_input} output {voxel_size_output}, Make sure to set the correct resolution for the input data."
            )
        self.channels = channels

    def check(self):
        """
        Checks if the checkpoint for the specified run and criterion exists.

        Raises:
            Exception
                If the checkpoint does not exist, an Exception is thrown which
                is logged and handled by training the model without head matching.
        Examples:
            >>> check()
        Notes:
            This function is called by the CosemStart class to check if the
            checkpoint for the specified run and criterion exists.
        """
        from dacapo.store.create_store import create_weights_store

        weights_store = create_weights_store()
        weights_dir = Path(weights_store.basedir, self.run, "checkpoints", "iterations")
        if not (weights_dir / self.criterion).exists():
            if not weights_dir.exists():
                weights_dir.mkdir(parents=True, exist_ok=True)
            path = weights_dir / self.criterion
            cosem.download_checkpoint(self.name, path)
        else:
            logger.info(f"Checkpoint for {self.name} exists.")

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
            >>> model = initialize_weights(model, new_head)
        Notes:
            This function is called by the CosemStart class to retrieve the weights
            from the dacapo store and load them into the model.
        """
        self.check()
        from dacapo.store.create_store import create_weights_store

        weights_store = create_weights_store()
        weights_dir = Path(weights_store.basedir, self.run, "checkpoints", "iterations")
        if not (weights_dir / self.criterion).exists():
            if not weights_dir.exists():
                weights_dir.mkdir(parents=True, exist_ok=True)
            path = weights_dir / self.criterion
            cosem.download_checkpoint(self.name, path)
        weights = weights_store._retrieve_weights(self.run, self.criterion)
        _set_weights(model, weights, self.run, self.criterion, self.channels, new_head)
