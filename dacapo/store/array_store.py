import zarr
import neuroglancer
import attr

from abc import ABC, abstractmethod
import itertools
import json
from upath import UPath as Path
from typing import Optional, Tuple


@attr.s
class LocalArrayIdentifier:
    """
    Represents a local array identifier.

    Attributes:
        container (Path): The path to the container.
        dataset (str): The dataset name.
    Method:
        __str__ : Returns the string representation of the identifier.
    """

    container: Path = attr.ib()
    dataset: str = attr.ib()


@attr.s
class LocalContainerIdentifier:
    """
    Represents a local container identifier.

    Attributes:
        container (Path): The path to the container.
    Method:
        array_identifier : Creates a local array identifier for the given dataset.

    """

    container: Path = attr.ib()

    def array_identifier(self, dataset) -> LocalArrayIdentifier:
        """
        Creates a local array identifier for the given dataset.

        Args:
            dataset: The dataset for which to create the array identifier.
        Returns:
            LocalArrayIdentifier: The local array identifier.
        Raises:
            TypeError: If the dataset is not a string.
        Examples:
            >>> container = Path('path/to/container')
            >>> container.array_identifier('dataset')
            LocalArrayIdentifier(container=Path('path/to/container'), dataset='dataset')
        """
        return LocalArrayIdentifier(self.container, dataset)


class ArrayStore(ABC):
    """
    Base class for array stores.

    Creates identifiers for the caller to create and write arrays. Provides
    only rudimentary support for IO itself (currently only to remove
    arrays).

    Attributes:
        container (Path): The path to the container.
        dataset (str): The dataset name.
    Method:
        __str__ : Returns the string representation of the identifier.

    """

    @abstractmethod
    def validation_prediction_array(
        self, run_name: str, iteration: int, dataset: str
    ) -> LocalArrayIdentifier:
        """
        Get the array identifier for a particular validation prediction.

        Args:
            run_name: The name of the run.
            iteration: The iteration number.
            dataset: The dataset name.
        Returns:
            LocalArrayIdentifier: The array identifier.
        Raises:
            NotImplementedError: If the method is not implemented.
        Examples:
            >>> validation_prediction_array('run_name', 1, 'dataset')
            LocalArrayIdentifier(container=Path('path/to/container'), dataset='dataset')
        """
        pass

    @abstractmethod
    def validation_output_array(
        self, run_name: str, iteration: int, parameters: str, dataset: str
    ) -> LocalArrayIdentifier:
        """
        Get the array identifier for a particular validation output.

        Args:
            run_name: The name of the run.
            iteration: The iteration number.
            parameters: The parameters.
            dataset: The dataset name.
        Returns:
            LocalArrayIdentifier: The array identifier.
        Raises:
            NotImplementedError: If the method is not implemented.
        Examples:
            >>> validation_output_array('run_name', 1, 'parameters', 'dataset')
            LocalArrayIdentifier(container=Path('path/to/container'), dataset='dataset')
        """
        pass

    @abstractmethod
    def validation_input_arrays(
        self, run_name: str, index: Optional[str] = None
    ) -> Tuple[LocalArrayIdentifier, LocalArrayIdentifier]:
        """
        Get an array identifiers for the validation input raw/gt.

        It would be nice to store raw/gt with the validation predictions/outputs.
        If we don't store these we would have to look up the datasplit config
        and figure out where to find the inputs for each run. If we write
        the data then we don't need to search for it.
        This convenience comes at the cost of some extra memory usage.

        Args:
            run_name: The name of the run.
            index: The index of the validation input.
        Returns:
            Tuple[LocalArrayIdentifier, LocalArrayIdentifier]: The array identifiers.
        Raises:
            NotImplementedError: If the method is not implemented.
        Examples:
            >>> validation_input_arrays('run_name', 'index')
            (LocalArrayIdentifier(container=Path('path/to/container'), dataset='dataset'), LocalArrayIdentifier(container=Path('path/to/container'), dataset='dataset'))

        """
        pass

    @abstractmethod
    def remove(self, array_identifier: "LocalArrayIdentifier") -> None:
        """
        Remove an array by its identifier.

        Args:
            array_identifier: The array identifier.
        Raises:
            NotImplementedError: If the method is not implemented.
        Examples:
            >>> remove(LocalArrayIdentifier(container=Path('path/to/container'), dataset='dataset'))

        """
        pass

    @abstractmethod
    def snapshot_container(self, run_name: str) -> LocalContainerIdentifier:
        """
        Get a container identifier for storage of a snapshot.

        Args:
            run_name: The name of the run.
        Returns:
            LocalContainerIdentifier: The container identifier.
        Raises:
            NotImplementedError: If the method is not implemented.
        Examples:
            >>> snapshot_container('run_name')
            LocalContainerIdentifier(container=Path('path/to/container'))
        """
        pass

    @abstractmethod
    def validation_container(self, run_name: str) -> LocalContainerIdentifier:
        """
        Get a container identifier for storage of a snapshot.

        Args:
            run_name: The name of the run.
        Returns:
            LocalContainerIdentifier: The container identifier.
        Raises:
            NotImplementedError: If the method is not implemented.
        Examples:
            >>> validation_container('run_name')
            LocalContainerIdentifier(container=Path('path/to/container'))
        """
        pass

    def _visualize_training(self, run):
        """
        Returns a neuroglancer link to visualize snapshots and validations.

        Args:
            run: The run.
        Returns:
            str: The neuroglancer link.
        Raises:
            NotImplementedError: If the method is not implemented.
        Examples:
            >>> _visualize_training(run)
            'http://neuroglancer-demo.appspot.com/#!{}'

        """
        # returns a neuroglancer link to visualize snapshots and validations
        snapshot_container = self.snapshot_container(run.name)
        validation_container = self.validation_container(run.name)
        snapshot_zarr = zarr.open(snapshot_container.container)
        validation_zarr = zarr.open(validation_container.container)

        snapshots = []
        validations = []

        def generate_groups(container):
            """
            Generate groups for snapshots and validations.

            Args:
                container: The container.
            Returns:
                function: The add_element function.
            Raises:
                NotImplementedError: If the method is not implemented.
            Examples:
                >>> generate_groups(container)
                function

            """

            def add_element(name, obj):
                """
                Add elements to the container.

                Args:
                    name: The name of the element.
                    obj: The object.
                Raises:
                    NotImplementedError: If the method is not implemented.
                Examples:
                    >>> add_element('name', 'obj')
                    None

                """
                if isinstance(obj, zarr.hierarchy.Array):
                    container.append(name)

            return add_element

        snapshot_zarr.visititems(
            lambda name, obj: generate_groups(snapshots)(name, obj)
        )
        validation_zarr.visititems(
            lambda name, obj: generate_groups(validations)(name, obj)
        )

        viewer = neuroglancer.Viewer()
        with viewer.txn() as s:
            snapshot_layers = {}
            for snapshot in snapshots:
                snapshot_layers[snapshot] = open_from_identifier(
                    snapshot_container.array_identifier(snapshot), name=snapshot
                )._neuroglancer_layer()

            validation_layers = {}
            for validation in validations:
                validation_layers[validation] = open_from_identifier(
                    validation_container.array_identifier(validation), name=validation
                )._neuroglancer_layer()

            for layer_name, (layer, kwargs) in itertools.chain(
                snapshot_layers.items(), validation_layers.items()
            ):
                s.layers.append(
                    name=layer_name,
                    layer=layer,
                    **kwargs,
                )

            s.layout = neuroglancer.row_layout(
                [
                    neuroglancer.LayerGroupViewer(layers=list(snapshot_layers.keys())),
                    neuroglancer.LayerGroupViewer(
                        layers=list(validation_layers.keys())
                    ),
                ]
            )
        return f"http://neuroglancer-demo.appspot.com/#!{json.dumps(viewer.state.to_json())}"
