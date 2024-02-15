from dacapo.experiments.datasplits.datasets.arrays.zarr_array import ZarrArray

import zarr
import neuroglancer
import attr

from abc import ABC, abstractmethod
import itertools
import json
from pathlib import Path
from typing import Optional, Tuple


@attr.s
class LocalArrayIdentifier:
    container: Path = attr.ib()
    dataset: str = attr.ib()


@attr.s
class LocalContainerIdentifier:
    container: Path = attr.ib()

    def array_identifier(self, dataset) -> LocalArrayIdentifier:
        return LocalArrayIdentifier(self.container, dataset)


class ArrayStore(ABC):
    """Base class for array stores.

    Creates identifiers for the caller to create and write arrays. Provides
    only rudimentary support for IO itself (currently only to remove
    arrays)."""

    @abstractmethod
    def validation_prediction_array(
        self, run_name: str, iteration: int, dataset: str
    ) -> LocalArrayIdentifier:
        """Get the array identifier for a particular validation prediction."""
        pass

    @abstractmethod
    def validation_output_array(
        self, run_name: str, iteration: int, parameters: str, dataset: str
    ) -> LocalArrayIdentifier:
        """Get the array identifier for a particular validation output."""
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
        """
        pass

    @abstractmethod
    def remove(self, array_identifier: 'LocalArrayIdentifier') -> None:
        """Remove an array by its identifier."""
        pass

    @abstractmethod
    def snapshot_container(self, run_name: str) -> LocalContainerIdentifier:
        """
        Get a container identifier for storage of a snapshot.
        """
        pass

    @abstractmethod
    def validation_container(self, run_name: str) -> LocalContainerIdentifier:
        """
        Get a container identifier for storage of a snapshot.
        """
        pass

    def _visualize_training(self, run):
        # returns a neuroglancer link to visualize snapshots and validations
        snapshot_container = self.snapshot_container(run.name)
        validation_container = self.validation_container(run.name)
        snapshot_zarr = zarr.open(snapshot_container.container)
        validation_zarr = zarr.open(validation_container.container)

        snapshots = []
        validations = []

        def generate_groups(container):
            def add_element(name, obj):
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
                snapshot_layers[snapshot] = ZarrArray.open_from_array_identifier(
                    snapshot_container.array_identifier(snapshot), name=snapshot
                )._neuroglancer_layer()

            validation_layers = {}
            for validation in validations:
                validation_layers[validation] = ZarrArray.open_from_array_identifier(
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
