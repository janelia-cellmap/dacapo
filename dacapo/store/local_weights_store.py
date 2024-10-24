from dacapo.experiments.datasplits.datasets.dataset import Dataset
from .weights_store import WeightsStore, Weights
from dacapo.experiments.run import Run

import torch

import json
from upath import UPath as Path
import logging
from typing import Optional, Union


logger = logging.getLogger(__name__)


class LocalWeightsStore(WeightsStore):
    """
    A local store for network weights.

    All weights are stored in a directory structure like this:

        ```
        basedir
        ├── run1
        │   ├── checkpoints
        │   │   ├── iterations
        │   │   │   ├── 0
        │   │   │   ├── 1
        │   │   │   ├── ...
        │   ├── dataset1
        │   │   ├── criterion1.json
        │   ├── dataset2
        │   │   ├── criterion2.json
        ├── run2
        │   ├── ...
        ```

    Attributes:
        basedir: The base directory where the weights are stored.
    Methods:
        latest_iteration: Return the latest iteration for which weights are available for the given run.
        store_weights: Store the network weights of the given run.
        retrieve_weights: Retrieve the network weights of the given run.
        remove: Remove the network weights of the given run.
        store_best: Store the best weights in a easy to find location.
        retrieve_best: Retrieve the best weights of the given run.
    Note:
        The weights are stored in the format of a Weights object, which is a simple container for the model and optimizer state dicts.

    """

    def __init__(self, basedir):
        """
        Create a new local weights store.

        Args:
            basedir: The base directory where the weights are stored.
        Raises:
            FileNotFoundError: If the directory does not exist.
        Examples:
            >>> store = LocalWeightsStore("weights")
        Note:
            The directory is created if it does not exist.

        """
        print(f"Creating local weights store in directory {basedir}")

        self.basedir = basedir

    def latest_iteration(self, run: str) -> Optional[int]:
        """
        Return the latest iteration for which weights are available for the
        given run.

        Args:
            run: The name of the run.
        Returns:
            The latest iteration for which weights are available, or None if no
            weights are available.
        Raises:
            FileNotFoundError: If the run directory does not exist.
        Examples:
            >>> store.latest_iteration("run1")
        Note:
            The iteration is determined by the number of the subdirectories in the "iterations" directory.
        """

        weights_dir = self.__get_weights_dir(run) / "iterations"

        iterations = sorted([int(path.parts[-1]) for path in weights_dir.glob("*")])

        if not iterations:
            return None

        return iterations[-1]

    def store_weights(self, run: Run, iteration: int):
        """
        Store the network weights of the given run.

        Args:
            run: The run object.
            iteration: The iteration number.
        Raises:
            FileNotFoundError: If the run directory does not exist.
        Examples:
            >>> store.store_weights(run, 0)
        Note:
            The weights are stored in the format of a Weights object, which is a simple container for the model and optimizer state dicts.
        """

        logger.warning(f"Storing weights for run {run}, iteration {iteration}")

        weights_dir = self.__get_weights_dir(run) / "iterations"
        weights_name = weights_dir / str(iteration)

        if not weights_dir.exists():
            weights_dir.mkdir(parents=True, exist_ok=True)

        weights = Weights(run.model.state_dict(), run.optimizer.state_dict())

        torch.save(weights, weights_name)

    def retrieve_weights(self, run: str, iteration: int) -> Weights:
        """
        Retrieve the network weights of the given run.

        Args:
            run: The name of the run.
            iteration: The iteration number.
        Returns:
            The network weights.
        Raises:
            FileNotFoundError: If the weights file does not exist.
        Examples:
            >>> store.retrieve_weights("run1", 0)
        Note:
            The weights are stored in the format of a Weights object, which is a simple container for the model and optimizer state dicts.
        """

        print(f"Retrieving weights for run {run}, iteration {iteration}")

        weights_name = self.__get_weights_dir(run) / "iterations" / str(iteration)

        weights: Weights = torch.load(
            weights_name, map_location="cpu", weights_only=False
        )
        if not isinstance(weights, Weights):
            # backwards compatibility
            weights = Weights(weights["model"], weights["optimizer"])

        return weights

    def _retrieve_weights(self, run: str, key: str) -> Weights:
        """
        Retrieves the weights for a given run and key.

        Args:
            run (str): The name of the run.
            key (str): The key of the weights.
        Returns:
            Weights: The retrieved weights.
        Raises:
            FileNotFoundError: If the weights file does not exist.
        Examples:
            >>> store._retrieve_weights("run1", "key1")
        Note:
            The weights are stored in the format of a Weights object, which is a simple container for the model and optimizer state dicts.
        """
        weights_name = self.__get_weights_dir(run) / key
        if not weights_name.exists():
            weights_name = self.__get_weights_dir(run) / "iterations" / key

        weights: Weights = torch.load(weights_name, map_location="cpu")
        if not isinstance(weights, Weights):
            # backwards compatibility
            weights = Weights(weights["model"], weights["optimizer"])

        return weights

    def remove(self, run: str, iteration: int):
        """
        Remove the weights for a specific run and iteration.

        Args:
            run (str): The name of the run.
            iteration (int): The iteration number.
        Raises:
            FileNotFoundError: If the weights file does not exist.
        Examples:
            >>> store.remove("run1", 0)
        Note:
            The weights are stored in the format of a Weights object, which is a simple container for the model and optimizer state dicts.
        """
        weights = self.__get_weights_dir(run) / "iterations" / str(iteration)
        weights.unlink()

    def store_best(self, run: str, iteration: int, dataset: str, criterion: str):
        """
        Store the best weights in a easy to find location.
        Symlinks weights from appropriate iteration
        # TODO: simply store a yaml of dataset/criterion -> iteration/parameter id

        Args:
            run (str): The name of the run.
            iteration (int): The iteration number.
            dataset (str): The name of the dataset.
            criterion (str): The criterion for selecting the best weights.
        Raises:
            FileNotFoundError: If the weights file does not exist.
        Examples:
            >>> store.store_best("run1", 0, "dataset1", "criterion1")
        Note:
            The best weights are stored in a json file that contains the iteration number.
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
        try:
            best_weights.symlink_to(iteration_weights)
        except FileExistsError:
            best_weights.unlink()
            best_weights.symlink_to(iteration_weights)

        with best_weights_json.open("w") as f:
            f.write(json.dumps({"iteration": iteration}))

    def retrieve_best(self, run: str, dataset: str | Dataset, criterion: str) -> int:
        """
        Retrieve the best weights for a given run, dataset, and criterion.

        Args:
            run (str): The name of the run.
            dataset (str | Dataset): The name of the dataset or a Dataset object.
            criterion (str): The criterion for selecting the best weights.
        Returns:
            int: The iteration number of the best weights.
        Raises:
            FileNotFoundError: If the weights file does not exist.
        Examples:
            >>> store.retrieve_best("run1", "dataset1", "criterion1")
        Note:
            The best weights are stored in a json file that contains the iteration number.
        """
        print(f"Retrieving weights for run {run}, criterion {criterion}")

        with (self.__get_weights_dir(run) / criterion / f"{dataset}.json").open(
            "r"
        ) as fd:
            weights_info = json.load(fd)

        return weights_info["iteration"]

    def _load_best(self, run: Run, criterion: str):
        """
        Load the best weights for a given run and criterion.

        Args:
            run (Run): The run for which to load the weights.
            criterion (str): The criterion for which to load the weights.
        Examples:
            >>> store._load_best(run, "criterion1")
        Note:
            This method is used internally by the store to load the best weights for a given run and criterion.
        """
        print(f"Retrieving weights for run {run}, criterion {criterion}")

        weights_name = self.__get_weights_dir(run) / f"{criterion}"

        weights: Weights = torch.load(weights_name, map_location="cpu")
        if not isinstance(weights, Weights):
            # backwards compatibility
            weights = Weights(weights["model"], weights["optimizer"])
        run.model.load_state_dict(weights.model)

    def __get_weights_dir(self, run: Union[str, Run]):
        """
        Get the directory path for storing weights checkpoints.

        Args:
            run: The name of the run or the run object.
        Returns:
            Path: The directory path for storing weights checkpoints.
        Raises:
            FileNotFoundError: If the run directory does not exist.
        Examples:
            >>> store.__get_weights_dir("run1")
        Note:
            The directory is created if it does not exist.
        """
        run = run if isinstance(run, str) else run.name

        return Path(self.basedir, run, "checkpoints")
