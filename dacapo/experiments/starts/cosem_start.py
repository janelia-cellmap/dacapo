from abc import ABC
import logging
from cellmap_models import cosem
from pathlib import Path
from .start import Start, _set_weights

logger = logging.getLogger(__file__)


def get_model_setup(run):
    
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
    

    def __init__(self, start_config):
        
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
