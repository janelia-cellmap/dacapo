from abc import ABC
import logging
from cellmap_models import cosem
from pathlib import Path
from .start import Start

logger = logging.getLogger(__file__)


def format_name(name):
    if "/" in name:
        run, criterion = name.split("/")
        return run, criterion
    else:
        raise ValueError(
            f"Invalid starter name format {name}. Must be in the format run/criterion"`
        )


class CosemStart(Start):
    def __init__(self, start_config):
        run, criterion = format_name(start_config.name)
        self.name = start_config.name
        super().__init__(run, criterion)

    def initialize_weights(self, model):
        from dacapo.store.create_store import create_weights_store

        weights_store = create_weights_store()
        weights_dir = Path(weights_store.basedir, self.run, "checkpoints", "iterations")
        if not (weights_dir / self.criterion).exists():
            if not weights_dir.exists():
                weights_dir.mkdir(parents=True, exist_ok=True)
            path = weights_dir / self.criterion
            cosem.download_checkpoint(self.name, path)
        weights = weights_store._retrieve_weights(self.run, self.criterion)
        super._set_weights(model, weights)
