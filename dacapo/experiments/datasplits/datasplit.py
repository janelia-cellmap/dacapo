from dacapo.experiments.datasplits.datasets import Dataset

import neuroglancer

from abc import ABC
from typing import List, Optional
import json
import itertools


class DataSplit(ABC):
    train: List[Dataset]
    validate: Optional[List[Dataset]]

    def _neuroglancer_link(self):
        viewer = neuroglancer.Viewer()
        with viewer.txn() as s:

            train_layers = {}
            for i, dataset in enumerate(self.train):
                train_layers.update(
                    dataset._neuroglancer_layers(
                        exclude_layers=set(train_layers.keys())
                    )
                )

            validate_layers = {}
            if self.validate is not None:
                for i, dataset in enumerate(self.validate):
                    validate_layers.update(
                        dataset._neuroglancer_layers(
                            exclude_layers=set(validate_layers.keys())
                        )
                    )

            for layer_name, (layer, kwargs) in itertools.chain(
                train_layers.items(), validate_layers.items()
            ):
                s.layers.append(
                    name=layer_name,
                    layer=layer,
                    **kwargs,
                )

            s.layout = neuroglancer.row_layout(
                [
                    neuroglancer.LayerGroupViewer(layers=list(train_layers.keys())),
                    neuroglancer.LayerGroupViewer(layers=list(validate_layers.keys())),
                ]
            )
        return f"http://neuroglancer-demo.appspot.com/#!{json.dumps(viewer.state.to_json())}"
