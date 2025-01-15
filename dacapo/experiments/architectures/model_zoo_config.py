import attr
import re

import torch
from .architecture import ArchitectureConfig

from funlib.geometry import Coordinate

from bioimageio.core import load_description_and_test
from bioimageio.core.model_adapters._pytorch_model_adapter import PytorchModelAdapter
from bioimageio.spec import InvalidDescr
from bioimageio.spec.model.v0_5 import (
    ModelDescr,
    OutputTensorDescr,
    InputTensorDescr,
)


@attr.s
class ModelZooConfig(ArchitectureConfig):
    """
    A thin wrapper allowing users to pass in any model zoo model
    """

    model_id: str = attr.ib(
        metadata={
            "help_text": "The model id from the model zoo to use. Can be any of:\n"
            '\t1) Url to a model zoo model (e.g. "https://.../rdf.yaml")\n'
            '\t2) Local path to a model zoo model (e.g. "some/local/rdf.yaml")\n'
            '\t3) Local path to a zipped model (e.g. "some/local/package.zip")\n'
            "\t4) Specific versioned model (e.g. {model_name}/{version})\n"
            "\t5) More options available, see: https://github.com/bioimage-io/spec-bioimage-io/tree/main"
        }
    )
    trainable_layers: str | None = attr.ib(None)

    _model_description: ModelDescr | None = None
    _model_adapter: PytorchModelAdapter | None = None

    def module(self) -> torch.nn.Module:
        module = self.model_adapter._network
        for name, param in module.named_parameters():
            if self.trainable_layers is not None and re.match(self.trainable_layers, name):
                param.requires_grad = True
            else:
                False
        return module

    @property
    def model_adapter(self) -> PytorchModelAdapter:
        if self._model_adapter is None:
            weights = self.model_description.weights

            return PytorchModelAdapter(
                outputs=self.model_description.outputs,
                weights=weights.pytorch_state_dict,
                devices=None,
            )
        return self._model_adapter

    @property
    def model_description(self) -> ModelDescr:
        if self._model_description is None:
            self._model_description = load_description_and_test(self.model_id)
            if isinstance(self._model_description, InvalidDescr):
                raise Exception("Invalid model description")
                self._model_description = self._model_description
        return self._model_description

    @property
    def input_desc(self) -> InputTensorDescr:
        assert len(self.model_description.inputs) == 1, (
            f"Only models with one input are supported, found {self.model_description.inputs}"
        )
        return self.model_description.inputs[0]

    @property
    def output_desc(self) -> OutputTensorDescr:
        assert len(self.model_description.outputs) == 1, (
            f"Only models with one output are supported, found {self.model_description.outputs}"
        )
        return self.model_description.outputs[0]

    @property
    def input_shape(self):
        shape = [
            axis.size.min
            for axis in self.input_desc.axes
            if axis.type not in ["batch", "channel", "index"]
        ]
        return Coordinate(shape)

    @property
    def num_in_channels(self) -> int:
        channel_axes = [axis for axis in self.input_desc.axes if axis.type == "channel"]
        assert len(channel_axes) == 1, (
            f"Only models with one input channel axis are supported, found {channel_axes}"
        )
        return channel_axes[0].size

    @property
    def num_out_channels(self) -> int:
        channel_axes = [
            axis for axis in self.output_desc.axes if axis.type == "channel"
        ]
        assert len(channel_axes) == 1, (
            f"Only models with one output channel axis are supported, found {channel_axes}"
        )
        return channel_axes[0].size

    def scale(self, input_voxel_size: Coordinate) -> Coordinate:
        # TODO: Implement scaling for model zoo models
        return input_voxel_size
