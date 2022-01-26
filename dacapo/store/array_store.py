from dacapo.experiments.datasplits.datasets.arrays.zarr_array import ZarrArray

import zarr
import neuroglancer

from abc import ABC, abstractmethod
import itertools
import json


class ArrayStore(ABC):
    """Base class for array stores.

    Creates identifiers for the caller to create and write arrays. Provides
    only rudimentary support for IO itself (currently only to remove
    arrays)."""

    @abstractmethod
    def validation_prediction_array(self, run_name, iteration):
        """Get the array identifier for a particular validation prediction."""
        pass

    @abstractmethod
    def validation_output_array(self, run_name, iteration, parameters):
        """Get the array identifier for a particular validation output."""
        pass

    @abstractmethod
    def remove(self, array_identifier):
        """Remove an array by its identifier."""
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
                    snapshot_container.array_container(snapshot), name=snapshot
                )._neuroglancer_layer()

            validation_layers = {}
            for validation in validations:
                validation_layers[validation] = ZarrArray.open_from_array_identifier(
                    validation_container.array_container(validation), name=validation
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
