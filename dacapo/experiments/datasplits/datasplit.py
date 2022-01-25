from dacapo import Options
from .datasets import Dataset

import neuroglancer

from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path
import json
import itertools


class DataSplit(ABC):
    @property
    @abstractmethod
    def train(self) -> List[Dataset]:
        """The Dataset to train on."""
        pass

    @property
    def validate(self) -> Optional[List[Dataset]]:
        return None

    def _neuroglancer_link(self):
        options = Options.instance()
        store_path = Path(options.runs_base_dir)

        viewer = neuroglancer.Viewer()
        with viewer.txn() as s:

            train_layers = {}
            for i, dataset in enumerate(self.train):
                train_layers.update(dataset._neuroglancer_layers(""))

            validate_layers = {}
            if self.validate is not None:
                for i, dataset in enumerate(self.validate):
                    validate_layers.update(dataset._neuroglancer_layers(""))

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
