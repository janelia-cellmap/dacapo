from .array_store import ArrayStore
from pathlib import Path
import attr
import logging
import shutil

logger = logging.getLogger(__name__)


@attr.s
class LocalContainerIdentifier:
    container: Path = attr.ib()

    def array_container(self, dataset):
        return LocalArrayIdentifier(self.container, dataset)


@attr.s
class LocalArrayIdentifier:

    container: Path = attr.ib()
    dataset: str = attr.ib()


class LocalArrayStore(ArrayStore):
    """A local array store that uses zarr containers."""

    def __init__(self, basedir):

        self.basedir = basedir

    def best_validation_array(self, run_name, criterion, index=None):
        container = self.validation_container(run_name).container
        if index is None:
            dataset = f"{criterion}"
        else:
            dataset = f"{index}/{criterion}"

        return LocalArrayIdentifier(container, dataset)

    def validation_prediction_array(self, run_name, iteration):
        """Get the array identifier for a particular validation prediction."""

        container = self.validation_container(run_name).container
        dataset = f"{iteration}/prediction"

        return LocalArrayIdentifier(container, dataset)

    def validation_output_array(self, run_name, iteration, parameters):
        """Get the array identifier for a particular validation output."""

        container = self.validation_container(run_name).container
        dataset = f"{iteration}/output/{parameters.id}"

        return LocalArrayIdentifier(container, dataset)

    def validation_input_arrays(self, run_name, index=None):
        """
        Get an array identifiers for the validation input raw/gt.

        It would be nice to store raw/gt with the validation predictions/outputs.
        If we don't store these we would have to look up the datasplit config
        and figure out where to find the inputs for each run. If we write
        the data then we don't need to search for it.
        This convenience comes at the cost of some extra memory usage.
        """

        container = self.validation_container(run_name).container
        if index is not None:
            dataset_prefix = f"inputs/{index}"
        else:
            dataset_prefix = "inputs"

        return LocalArrayIdentifier(
            container, f"{dataset_prefix}/raw"
        ), LocalArrayIdentifier(container, f"{dataset_prefix}/gt")

    def snapshot_container(self, run_name):
        """
        Get a container identifier for storage of a snapshot.
        """
        return LocalContainerIdentifier(
            Path(self.__get_run_dir(run_name), "snapshot.zarr")
        )

    def validation_container(self, run_name):
        """
        Get a container identifier for storage of a snapshot.
        """
        return LocalContainerIdentifier(
            Path(self.__get_run_dir(run_name), "validation.zarr")
        )

    def remove(self, array_identifier):

        container = array_identifier.container
        dataset = array_identifier.dataset

        assert container.suffix == ".zarr", (
            "The container path does not end with '.zarr'. Stopping here to "
            "prevent data loss."
        )

        path = Path(container, dataset)

        if not path.exists():
            logger.warning(
                "Asked to remove dataset %s in container %s, but doesn't " "exist.",
                dataset,
                container,
            )
            return

        if not path.is_dir():
            logger.warning(
                "Asked to remove dataset %s in container %s, but is not "
                "a directory. Will not delete.",
                dataset,
                container,
            )
            return

        logger.info("Removing dataset %s in container %s", dataset, container)
        shutil.rmtree(path)

    def __get_run_dir(self, run_name):

        return Path(self.basedir, run_name)
