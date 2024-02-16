from .array_store import ArrayStore, LocalArrayIdentifier, LocalContainerIdentifier

from pathlib import Path
import logging
import shutil
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class LocalArrayStore(ArrayStore):
    """
    A class that manages a local array store using zarr containers.

    Attributes:
        basedir: Directory to store the local array.

    """

    def __init__(self, basedir):
        """
        Initialize the LocalArrayStore with base directory.

        Args:
            basedir: Directory to store the local array.
        """
        self.basedir = basedir

    def best_validation_array(
        self, run_name: str, criterion: str, index: Optional[str] = None
    ) -> LocalArrayIdentifier:
        """
        Get the best validation array for given criterion and index.

        Args:
            run_name: Name of the run.
            criterion: Criteria to choose the best validation.
            index: Index to look for the best validation.

        Returns:
            An instance of LocalArrayIdentifier.
        """
        container = self.validation_container(run_name).container
        if index is None:
            dataset = f"{criterion}"
        else:
            dataset = f"{index}/{criterion}"

        return LocalArrayIdentifier(container, dataset)

    def validation_prediction_array(
        self, run_name: str, iteration: int, dataset: str
    ) -> LocalArrayIdentifier:
        """
        Get the array identifier for a particular validation prediction.

        Args:
            run_name: Name of the run.
            iteration: Iteration count of the validation prediction.
            dataset: Dataset to look for the validation prediction.

        Returns:
            An instance of LocalArrayIdentifier.
        """
        container = self.validation_container(run_name).container
        dataset = f"{iteration}/{dataset}/prediction"

        return LocalArrayIdentifier(container, dataset)

    def validation_output_array(
        self, run_name: str, iteration: int, parameters: str, dataset: str
    ) -> LocalArrayIdentifier:
        """
        Get the array identifier for a particular validation output.

        Args:
            run_name: Name of the run.
            iteration: Iteration count of the validation output.
            parameters: Parameters of the validation.
            dataset: Dataset to look for the validation output.

        Returns:
            An instance of LocalArrayIdentifier.
        """
        container = self.validation_container(run_name).container
        dataset = f"{iteration}/{dataset}/output/{parameters}"

        return LocalArrayIdentifier(container, dataset)

    def validation_input_arrays(
        self, run_name: str, index: Optional[str] = None
    ) -> Tuple[LocalArrayIdentifier, LocalArrayIdentifier]:
        """
        Get an array identifiers for the validation input raw/gt.

        Args:
            run_name: Name of the run.
            index: Index to look for the validation inputs.

        Returns:
            A tuple containing instances of LocalArrayIdentifier for raw and gt.
        """
        container = self.validation_container(run_name).container
        if index is not None:
            dataset_prefix = f"inputs/{index}"
        else:
            dataset_prefix = "inputs"

        return (
            LocalArrayIdentifier(container, f"{dataset_prefix}/raw"),
            LocalArrayIdentifier(container, f"{dataset_prefix}/gt"),
        )

    def snapshot_container(self, run_name: str) -> LocalContainerIdentifier:
        """
        Get a container identifier for storage of a snapshot.

        Args:
            run_name: Name of the run.

        Returns:
            An instance of LocalContainerIdentifier.
        """
        return LocalContainerIdentifier(
            Path(self.__get_run_dir(run_name), "snapshot.zarr")
        )

    def validation_container(self, run_name: str) -> LocalContainerIdentifier:
        """
        Get a container identifier for storage of a snapshot.

        Args:
            run_name: Name of the run.

        Returns:
            An instance of LocalContainerIdentifier.
        """
        return LocalContainerIdentifier(
            Path(self.__get_run_dir(run_name), "validation.zarr")
        )

    def remove(self, array_identifier: "LocalArrayIdentifier") -> None:
        """
        Remove a dataset in a container.

        Args:
            array_identifier: LocalArrayIdentifier to specify the dataset and the container.

        Raises:
            AssertionError: If the container path does not end with '.zarr'.
        """
        container = array_identifier.container
        dataset = array_identifier.dataset

        assert container.suffix == ".zarr", (
            "The container path does not end with '.zarr'. Stopping here to "
            "prevent data loss."
        )

        path = Path(container, dataset)

        if not path.exists():
            logger.warning(
                "Asked to remove dataset %s in container %s, but it doesn't exist.",
                dataset,
                container,
            )
            return

        if not path.is_dir():
            logger.warning(
                "Asked to remove dataset %s in container %s, but it is not "
                "a directory. Will not delete.",
                dataset,
                container,
            )
            return

        logger.info("Removing dataset %s in container %s", dataset, container)
        shutil.rmtree(path)

    def __get_run_dir(self, run_name: str) -> Path:
        """
        Get the directory path for a run.

        Args:
            run_name: Name of the run.

        Returns:
            A pathlib.Path object representing the run directory.
        """
        return Path(self.basedir, run_name)