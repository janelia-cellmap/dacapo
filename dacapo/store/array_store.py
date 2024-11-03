from dacapo.experiments.datasplits.datasets.arrays.zarr_array import ZarrArray

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
    

    container: Path = attr.ib()
    dataset: str = attr.ib()


@attr.s
class LocalContainerIdentifier:
    

    container: Path = attr.ib()

    def array_identifier(self, dataset) -> LocalArrayIdentifier:
        
        return LocalArrayIdentifier(self.container, dataset)


class ArrayStore(ABC):
    

    @abstractmethod
    def validation_prediction_array(
        self, run_name: str, iteration: int, dataset: str
    ) -> LocalArrayIdentifier:
        
        pass

    @abstractmethod
    def validation_output_array(
        self, run_name: str, iteration: int, parameters: str, dataset: str
    ) -> LocalArrayIdentifier:
        
        pass

    @abstractmethod
    def validation_input_arrays(
        self, run_name: str, index: Optional[str] = None
    ) -> Tuple[LocalArrayIdentifier, LocalArrayIdentifier]:
        
        pass

    @abstractmethod
    def remove(self, array_identifier: "LocalArrayIdentifier") -> None:
        
        pass

    @abstractmethod
    def snapshot_container(self, run_name: str) -> LocalContainerIdentifier:
        
        pass

    @abstractmethod
    def validation_container(self, run_name: str) -> LocalContainerIdentifier:
        
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
