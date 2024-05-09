from .array_store import ArrayStore, LocalArrayIdentifier, LocalContainerIdentifier

from upath import UPath as Path
import logging
import shutil
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class LocalArrayStore(ArrayStore):
    """
    A local array store that uses zarr containers.

    Attributes:
        basedir: The base directory where the store will write data.
    Methods:
        best_validation_array: Get the array identifier for the best validation array.
        validation_prediction_array: Get the array identifier for a particular validation prediction.
        validation_output_array: Get the array identifier for a particular validation output.
        validation_input_arrays: Get the array identifiers for the validation input raw/gt.
        snapshot_container: Get a container identifier for storage of a snapshot.
        validation_container: Get a container identifier for storage of a validation.
        remove: Remove a dataset from a container.

    """

    def __init__(self, basedir):
        """
        Initialize the LocalArrayStore.

        Args:
            basedir (str): The base directory where the store will write data.
        Raises:
            ValueError: If the basedir is not a directory.
        Examples:
            >>> store = LocalArrayStore("/path/to/store")

        """
        self.basedir = basedir

    def best_validation_array(
        self, run_name: str, criterion: str, index: Optional[str] = None
    ) -> LocalArrayIdentifier:
        """
        Get the array identifier for the best validation array.

        Args:
            run_name (str): The name of the run.
            criterion (str): The criterion for the validation array.
            index (str, optional): The index of the validation array. Defaults to None.
        Returns:
            LocalArrayIdentifier: The array identifier for the best validation array.
        Raises:
            ValueError: If the container does not exist.
        Examples:
            >>> store.best_validation_array("run1", "loss")

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
            run_name (str): The name of the run.
            iteration (int): The iteration of the validation prediction.
            dataset (str): The dataset of the validation prediction.
        Returns:
            LocalArrayIdentifier: The array identifier for the validation prediction.
        Raises:
            ValueError: If the container does not exist.
        Examples:
            >>> store.validation_prediction_array("run1", 0, "train")
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
            run_name (str): The name of the run.
            iteration (int): The iteration of the validation output.
            parameters (str): The parameters of the validation output.
            dataset (str): The dataset of the validation output.
        Returns:
            LocalArrayIdentifier: The array identifier for the validation output.
        Raises:
            ValueError: If the container does not exist.
        Examples:
            >>> store.validation_output_array("run1", 0, "params1", "train")
        """
        container = self.validation_container(run_name).container
        dataset = f"{iteration}/{dataset}/output/{parameters}"
        return LocalArrayIdentifier(container, dataset)

    def validation_input_arrays(
        self, run_name: str, index: Optional[str] = None
    ) -> Tuple[LocalArrayIdentifier, LocalArrayIdentifier]:
        """
        Get the array identifiers for the validation input raw/gt.

        It would be nice to store raw/gt with the validation predictions/outputs.
        If we don't store these we would have to look up the datasplit config
        and figure out where to find the inputs for each run. If we write
        the data then we don't need to search for it.
        This convenience comes at the cost of some extra memory usage.

        Args:
            run_name (str): The name of the run.
            index (str, optional): The index of the validation input. Defaults to None.
        Returns:
            Tuple[LocalArrayIdentifier, LocalArrayIdentifier]: The array identifiers for the validation input raw/gt.
        Raises:
            ValueError: If the container does not exist.
        Examples:
            >>> store.validation_input_arrays("run1")
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
            run_name (str): The name of the run.
        Returns:
            LocalContainerIdentifier: The container identifier for the snapshot.
        Raises:
            ValueError: If the container does not exist.
        Examples:
            >>> store.snapshot_container("run1")

        """

        return LocalContainerIdentifier(
            Path(self.__get_run_dir(run_name), "snapshot.zarr")
        )

    def validation_container(self, run_name: str) -> LocalContainerIdentifier:
        """
        Get a container identifier for storage of a validation.

        Args:
            run_name (str): The name of the run.
        Returns:
            LocalContainerIdentifier: The container identifier for the validation.
        Raises:
            ValueError: If the container does not exist.
        Examples:
            >>> store.validation_container("run1")

        """

        return LocalContainerIdentifier(
            Path(self.__get_run_dir(run_name), "validation.zarr")
        )

    def remove(self, array_identifier: "LocalArrayIdentifier") -> None:
        """
        Remove a dataset from a container.

        Args:
            array_identifier (LocalArrayIdentifier): The array identifier of the dataset to remove.
        Raises:
            ValueError: If the container path does not end with '.zarr'.
        Examples:
            >>> store.remove(array_identifier)

        """
        container = array_identifier.container

        dataset = array_identifier.dataset

        assert container.suffix == ".zarr", (
            f"The container path does not end with '.zarr'. Stopping here to "
            f"prevent data loss."
        )
        path = Path(container, dataset)

        if not path.exists():
            logger.warning(
                f"Asked to remove dataset {dataset} in container {container}, but it doesn't exist."
            )
            return

        if not path.is_dir():
            logger.warning(
                f"Asked to remove dataset {dataset} in container {container}, but it is not a directory. Will not delete."
            )
            return
        print(f"Removing dataset {dataset} in container {container}")

        shutil.rmtree(path)

    def __get_run_dir(self, run_name: str) -> Path:
        """
        Get the directory path for a run.

        Args:
            run_name (str): The name of the run.
        Returns:
            Path: The directory path for the run.
        Raises:
            ValueError: If the run directory does not exist.
        Examples:
            >>> store.__get_run_dir("run1")
        """
        return Path(self.basedir, run_name)
