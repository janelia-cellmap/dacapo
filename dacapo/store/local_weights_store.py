from .weights_store import WeightsStore
from pathlib import Path
import logging
import torch

logger = logging.getLogger(__name__)


class LocalWeightsStore(WeightsStore):
    """A local store for network weights."""

    def __init__(self, basedir):

        logger.info("Creating local weights store in directory %s", basedir)

        self.basedir = basedir

    def latest_iteration(self, run):
        """Return the latest iteration for which weights are available for the
        given run."""

        weights_dir = self.__get_weights_dir(run)

        iterations = sorted([
            int(path.parts[-1])
            for path in weights_dir.glob('*')
        ])

        if not iterations:
            return None

        return iterations[-1]

    def store_weights(self, run, iteration):
        """Store the network weights of the given run."""

        logger.info(
            "Storing weights for run %s, iteration %d",
            run.name,
            iteration)

        weights_dir = self.__get_weights_dir(run)
        weights_name = Path(weights_dir, str(iteration))

        if not weights_dir.exists():
            weights_dir.mkdir(parents=True, exist_ok=True)

        weights = {
            'model': run.model.state_dict(),
            'optimizer': run.optimizer.state_dict()
        }

        torch.save(weights, weights_name)
    def store_best(self, run, iteration, criterion):
        """
        Take the weights from run/iteration and store it
        in run/criterion.
        """

        # must exist since we must read run/iteration weights
        weights_dir = self.__get_weights_dir(run)
        iteration_weights = Path(weights_dir, f"{iteration}")
        best_weights = Path(weights_dir, criterion)

        best_weights.write_bytes(iteration_weights.read_bytes())

    def retrieve_weights(self, run, iteration):
        """Retrieve the network weights of the given run."""

        logger.info(
            "Retrieving weights for run %s, iteration %d",
            run.name,
            iteration)

        weights_dir = self.__get_weights_dir(run)
        weights_name = Path(weights_dir, str(iteration))

        weights = torch.load(weights_name)

        run.model.load_state_dict(weights['model'])
        run.optimizer.load_state_dict(weights['optimizer'])

    def __get_weights_dir(self, run):

        return Path(self.basedir, run.name, 'checkpoints')
