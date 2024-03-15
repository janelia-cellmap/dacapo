from .datasets import Dataset

import neuroglancer
from funlib.show.neuroglancer import add_layer
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
                    dataset._neuroglancer_sources(
                        # exclude_layers=set(train_layers.keys())
                    )
                )

            validate_layers = {}
            if self.validate is not None:
                for i, dataset in enumerate(self.validate):
                    validate_layers.update(
                        dataset._neuroglancer_sources(
                            # exclude_layers=set(validate_layers.keys())
                        )
                    )

            for k,elms in itertools.chain(
                train_layers.items(), validate_layers.items()
            ):
                if type(elms) is list:
                    elms = elms[0]
                layer, layer_name = elms
                add_layer(
                    context=s,
                    array = layer,
                    name=k,
                )


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
