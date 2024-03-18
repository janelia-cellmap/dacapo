from .datasets import Dataset
import neuroglancer
from abc import ABC
from typing import List, Optional
import json
import itertools


class DataSplit(ABC):
    train: List[Dataset]
    validate: Optional[List[Dataset]]

    def _neuroglancer(self,embedded = False):
        neuroglancer.set_server_bind_address('0.0.0.0')
        viewer = neuroglancer.Viewer()
        with viewer.txn() as s:
            train_layers = {}
            for i, dataset in enumerate(self.train):
                train_layers.update(
                    dataset._neuroglancer_layers(
                        # exclude_layers=set(train_layers.keys())
                    )
                )

            validate_layers = {}
            if self.validate is not None:
                for i, dataset in enumerate(self.validate):
                    validate_layers.update(
                        dataset._neuroglancer_layers(
                            # exclude_layers=set(validate_layers.keys())
                        )
                    )

            for k,layer in itertools.chain(
                train_layers.items(), validate_layers.items()
            ):
                s.layers[k]= layer

            s.layout = neuroglancer.row_layout(
                [
                    neuroglancer.LayerGroupViewer(layers=list(train_layers.keys())),
                    neuroglancer.LayerGroupViewer(layers=list(validate_layers.keys())),
                ]
            )
        print(f"Neuroglancer link: {viewer}")
        if embedded:
            from IPython.display import IFrame
            return IFrame(viewer.get_viewer_url(), width=800, height=600)
        return viewer
