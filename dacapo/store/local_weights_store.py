from .weights_store import WeightsStore

import torch

import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class LocalWeightsStore(WeightsStore):
    """A local store for network weights."""

    def __init__(self, basedir):

        logger.info("Creating local weights store in directory %s", basedir)

        self.basedir = basedir

    def latest_iteration(self, run):
        """Return the latest iteration for which weights are available for the
        given run."""

        weights_dir = self.__get_weights_dir(run) / "iterations"

        iterations = sorted([int(path.parts[-1]) for path in weights_dir.glob("*")])

        if not iterations:
            return None

        return iterations[-1]

    def store_weights(self, run, iteration, remove_old=False):
        """Store the network weights of the given run."""

        logger.info("Storing weights for run %s, iteration %d", run.name, iteration)

        weights_dir = self.__get_weights_dir(run) / "iterations"
        weights_name = weights_dir / str(iteration)

        if not weights_dir.exists():
            weights_dir.mkdir(parents=True, exist_ok=True)

        weights = {
            "model": run.model.state_dict(),
            "optimizer": run.optimizer.state_dict(),
        }

        if remove_old:
            for checkpoint in list(weights_dir.iterdir()):
                if int(checkpoint.name) < iteration:
                    self.remove(run, int(checkpoint.name))

        torch.save(weights, weights_name)

    def remove(self, run, iteration):
        weights = self.__get_weights_dir(run) / "iterations" / str(iteration)
        weights.unlink()

    def store_best(self, run, iteration, criterion):
        """
        Take the weights from run/iteration and store it
        in run/criterion.
        """

        # must exist since we must read run/iteration weights
        weights_dir = self.__get_weights_dir(run)
        iteration_weights = weights_dir / "iterations" / f"{iteration}"
        best_weights = weights_dir / criterion

        best_weights.write_bytes(iteration_weights.read_bytes())
        with (weights_dir / f"{criterion}.json").open("w") as f:
            f.write(json.dumps({"iteration": iteration}))

    def retrieve_best(self, run, criterion):
        run_name = run if isinstance(run, str) else run.name

        logger.info("Retrieving weights for run %s, criterion %s", run_name, criterion)

        weights_name = self.__get_weights_dir(run) / criterion

        weights = torch.load(weights_name, map_location="cpu")

        if isinstance(run, str):
            return weights
        else:
            # load the model weights
            run.model.load_state_dict(weights["model"])
            run.optimizer.load_state_dict(weights["optimizer"])

    def retrieve_weights(self, run, iteration):
        """Retrieve the network weights of the given run."""

        logger.info("Retrieving weights for run %s, iteration %d", run.name, iteration)

        weights_name = self.__get_weights_dir(run) / "iterations" / str(iteration)

        weights = torch.load(weights_name, map_location="cpu")

        run.model.load_state_dict(weights["model"])
        run.optimizer.load_state_dict(weights["optimizer"])

    def __get_weights_dir(self, run):
        run = run if isinstance(run, str) else run.name

        return Path(self.basedir, run, "checkpoints")
