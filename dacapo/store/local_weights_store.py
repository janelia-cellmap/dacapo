from .weights_store import WeightsStore, Weights
from dacapo.experiments.run import Run

import torch

import json
from pathlib import Path
import logging
from typing import Optional, Union


logger = logging.getLogger(__name__)


class LocalWeightsStore(WeightsStore):
    """A local store for network weights."""

    def __init__(self, basedir):
        logger.info("Creating local weights store in directory %s", basedir)

        self.basedir = basedir

    def latest_iteration(self, run: str) -> Optional[int]:
        """Return the latest iteration for which weights are available for the
        given run."""

        weights_dir = self.__get_weights_dir(run) / "iterations"

        iterations = sorted([int(path.parts[-1]) for path in weights_dir.glob("*")])

        if not iterations:
            return None

        return iterations[-1]

    def store_weights(self, run: Run, iteration: int):
        """Store the network weights of the given run."""

        logger.warning("Storing weights for run %s, iteration %d", run, iteration)

        weights_dir = self.__get_weights_dir(run) / "iterations"
        weights_name = weights_dir / str(iteration)

        if not weights_dir.exists():
            weights_dir.mkdir(parents=True, exist_ok=True)

        weights = Weights(run.model.state_dict(), run.optimizer.state_dict())

        torch.save(weights, weights_name)

    def retrieve_weights(self, run: str, iteration: int) -> Weights:
        """Retrieve the network weights of the given run."""

        logger.info("Retrieving weights for run %s, iteration %d", run, iteration)

        weights_name = self.__get_weights_dir(run) / "iterations" / str(iteration)

        weights: Weights = torch.load(weights_name, map_location="cpu")
        if not isinstance(weights, Weights):
            # backwards compatibility
            weights = Weights(weights["model"], weights["optimizer"])

        return weights

    def _retrieve_weights(self, run: str, key: str) -> Weights:
        weights_name = self.__get_weights_dir(run) / key
        if not weights_name.exists():
            weights_name = self.__get_weights_dir(run) / "iterations" / key

        weights: Weights = torch.load(weights_name, map_location="cpu")
        if not isinstance(weights, Weights):
            # backwards compatibility
            weights = Weights(weights["model"], weights["optimizer"])

        return weights

    def remove(self, run: str, iteration: int):
        weights = self.__get_weights_dir(run) / "iterations" / str(iteration)
        weights.unlink()

    def store_best(self, run: str, iteration: int, dataset: str, criterion: str):
        """
        Store the best weights in a easy to find location.
        Symlinks weights from appropriate iteration
        # TODO: simply store a toml of dataset/criterion -> iteration/parameter id
        """

        # must exist since we must read run/iteration weights
        weights_dir = self.__get_weights_dir(run)
        iteration_weights = weights_dir / "iterations" / f"{iteration}"
        best_weights = weights_dir / dataset / criterion
        best_weights_json = weights_dir / dataset / f"{criterion}.json"

        if not best_weights.parent.exists():
            best_weights.parent.mkdir(parents=True)

        if best_weights.exists():
            best_weights.unlink()
        best_weights.symlink_to(iteration_weights)
        with best_weights_json.open("w") as f:
            f.write(json.dumps({"iteration": iteration}))

    def retrieve_best(self, run: str, dataset: str, criterion: str) -> int:
        logger.info("Retrieving weights for run %s, criterion %s", run, criterion)

        with (self.__get_weights_dir__(run) / criterion / f"{dataset}.json").open("r") as fd:
            weights_info = json.load(fd)

        return weights_info["iteration"]

    def _load_best(self, run: Run, criterion: str):
        logger.info("Retrieving weights for run %s, criterion %s", run, criterion)

        weights_name = self.__get_weights_dir(run) / f"{criterion}"

        weights: Weights = torch.load(weights_name, map_location="cpu")
        if not isinstance(weights, Weights):
            # backwards compatibility
            weights = Weights(weights["model"], weights["optimizer"])
        run.model.load_state_dict(weights.model)

    def __get_weights_dir(self, run: Union[str, Run]):
        run = run if isinstance(run, str) else run.name

        return Path(self.basedir, run, "checkpoints")
