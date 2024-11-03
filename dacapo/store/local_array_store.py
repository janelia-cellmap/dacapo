from .array_store import ArrayStore, LocalArrayIdentifier, LocalContainerIdentifier

from upath import UPath as Path
import logging
import shutil
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class LocalArrayStore(ArrayStore):
    

    def __init__(self, basedir):
        
        self.basedir = basedir

    def best_validation_array(
        self, run_name: str, criterion: str, index: Optional[str] = None
    ) -> LocalArrayIdentifier:
        

        container = self.validation_container(run_name).container
        if index is None:
            dataset = f"{criterion}"
        else:
            dataset = f"{index}/{criterion}"
        return LocalArrayIdentifier(container, dataset)

    def validation_prediction_array(
        self, run_name: str, iteration: int, dataset: str
    ) -> LocalArrayIdentifier:
        
        container = self.validation_container(run_name).container
        dataset = f"{iteration}/{dataset}/prediction"
        return LocalArrayIdentifier(container, dataset)

    def validation_output_array(
        self, run_name: str, iteration: int, parameters: str, dataset: str
    ) -> LocalArrayIdentifier:
        
        container = self.validation_container(run_name).container
        dataset = f"{iteration}/{dataset}/output/{parameters}"
        return LocalArrayIdentifier(container, dataset)

    def validation_input_arrays(
        self, run_name: str, index: Optional[str] = None
    ) -> Tuple[LocalArrayIdentifier, LocalArrayIdentifier]:
        
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
        

        return LocalContainerIdentifier(
            Path(self.__get_run_dir(run_name), "snapshot.zarr")
        )

    def validation_container(self, run_name: str) -> LocalContainerIdentifier:
        

        return LocalContainerIdentifier(
            Path(self.__get_run_dir(run_name), "validation.zarr")
        )

    def remove(self, array_identifier: "LocalArrayIdentifier") -> None:
        
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
        
        return Path(self.basedir, run_name)
