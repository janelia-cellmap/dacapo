from .datasets import Dataset

import neuroglancer

from abc import ABC
from typing import List, Optional
from funlib.show.neuroglancer import add_layer
import json
import itertools


class DataSplit(ABC):
    train: List[Dataset]
    validate: Optional[List[Dataset]]

    def _neuroglancer_link(self):
        neuroglancer.set_server_bind_address('0.0.0.0')
        viewer = neuroglancer.Viewer()
        with viewer.txn() as s:
            train_layers = {}
            for i, dataset in enumerate(self.train):

                train_layers.update(
                    dataset._neuroglancer_sources(
                        exclude_layers=set(train_layers.keys())
                    )
                )

            validate_layers = {}
            if self.validate is not None:
                for i, dataset in enumerate(self.validate):
                    validate_layers.update(
                        dataset._neuroglancer_sources(
                            exclude_layers=set(validate_layers.keys())
                        )
                    )

            for elms in itertools.chain(
                train_layers.values(), validate_layers.values()
            ):
                if type(elms) is list:
                    elms = elms[0]
                layer, layer_name = elms
                add_layer(context=s,
                          array=layer,
                          name=layer_name)
                # s.layers.append(
                #     name=layer_name,
                #     layer=layer,
                #     **kwargs,
                # )

            s.layout = neuroglancer.row_layout(
                [
                    neuroglancer.LayerGroupViewer(layers=list(train_layers.keys())),
                    neuroglancer.LayerGroupViewer(layers=list(validate_layers.keys())),
                ]
            )
        url = str(viewer)
        print("Neuroglancer link:", url)
        return viewer
        # return f"http://neuroglancer-demo.appspot.com/#!{json.dumps(viewer.state.to_json())}"
