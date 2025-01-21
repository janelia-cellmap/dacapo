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

from pathlib import Path
import zipfile
import numpy as np


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
    trainable_layers: str | None = attr.ib(
        default=None, metadata={"help_text": "Regex pattern for trainable layers"}
    )

    _model_description: ModelDescr | None = None
    _model_adapter: PytorchModelAdapter | None = None

    def module(self) -> torch.nn.Module:
        module = self.model_adapter._network
        for name, param in module.named_parameters():
            if self.trainable_layers is not None and re.match(
                self.trainable_layers, name
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False
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
            if isinstance(self.model_id, Path) and self.model_id.suffix == ".zip":
                with zipfile.ZipFile(self.model_id, "r") as zip_ref:
                    zip_ref.extractall(Path(f"{self.model_id}.unzip"))
                self._model_description = load_description_and_test(
                    Path(f"{self.model_id}.unzip")
                )
            else:
                self._model_description = load_description_and_test(self.model_id)
            if isinstance(self._model_description, InvalidDescr):
                raise Exception("Invalid model description")
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
        input_axes = [
            axis
            for axis in self.input_desc.axes
            if axis.type not in ["batch", "channel", "index"]
        ]
        output_axes = [
            axis
            for axis in self.output_desc.axes
            if axis.type not in ["batch", "channel", "index"]
        ]
        assert all(
            [
                in_axis.id == out_axis.id
                for in_axis, out_axis in zip(input_axes, output_axes)
            ]
        )
        scale = np.array(
            [
                in_axis.scale / out_axis.scale
                for in_axis, out_axis in zip(input_axes, output_axes)
            ]
        )
        output_voxel_size = Coordinate(np.array(input_voxel_size) / scale)
        return output_voxel_size
