from typing import Optional
import neuroglancer
from IPython.display import IFrame
import numpy as np
import gunpowder as gp
from funlib.persistence import Array
from dacapo.experiments.datasplits.datasets.arrays import ZarrArray
from funlib.persistence import open_ds
from threading import Thread
import neuroglancer
from neuroglancer.viewer_state import ViewerState
import os
from dacapo.experiments.run import Run
from dacapo.store.create_store import create_array_store
from IPython.display import IFrame
import time
import copy
import json


def get_viewer(
    arrays: dict, width: int = 1500, height: int = 600, headless: bool = True
) -> neuroglancer.Viewer | IFrame:
    for name, array_data in arrays.items():
        array = array_data["array"]
        if hasattr(array, "to_ndarray"):
            arrays[name]["array"] = array.to_ndarray()
        else:
            arrays[name]["array"] = array.data
        if hasattr(array, "voxel_size"):
            arrays[name]["voxel_sizes"] = array.voxel_size
        else:
            arrays[name]["voxel_sizes"] = array.spec.voxel_size

    neuroglancer.set_server_bind_address("0.0.0.0")
    viewer = neuroglancer.Viewer()
    with viewer.txn() as state:
        state.showSlices = False
        for name, array_data in arrays.items():
            meshes = "meshes" in array_data and array_data["meshes"]
            is_seg = meshes or ("is_seg" in array_data and array_data["is_seg"])
            if is_seg:
                add_seg_layer(
                    state, name, array_data["array"], array_data["voxel_sizes"], meshes
                )
            else:
                add_scalar_layer(
                    state, name, array_data["array"], array_data["voxel_sizes"]
                )

    if headless:
        return viewer
    else:
        return IFrame(src=viewer, width=width, height=height)


def add_seg_layer(state, name, data, voxel_size, meshes=False):
    if meshes:
        kwargs = {"segments": np.unique(data[data > 0])}
    else:
        kwargs = {}
    state.layers[name] = neuroglancer.SegmentationLayer(
        # segments=[str(i) for i in np.unique(data[data > 0])], # this line will cause all objects to be selected and thus all meshes to be generated...will be slow if lots of high res meshes
        source=neuroglancer.LocalVolume(
            data=data,
            dimensions=neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                units=["nm", "nm", "nm"],
                scales=voxel_size,
            ),
        ),
        **kwargs,
    )


def add_scalar_layer(state, name, data, voxel_size):
    state.layers[name] = neuroglancer.ImageLayer(
        source=neuroglancer.LocalVolume(
            data=data,
            dimensions=neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                units=["nm", "nm", "nm"],
                scales=voxel_size,
            ),
        ),
    )


class NeuroglancerRunViewer:
    def __init__(self, run: Run):
        self.run: Run = run
        self.most_recent_iteration = 0
        self.prediction = None

    def updated_neuroglancer_layer(self, layer_name, ds):
        source = neuroglancer.LocalVolume(
            data=ds.data,
            dimensions=neuroglancer.CoordinateSpace(
                names=["c", "z", "y", "x"],
                units=["", "nm", "nm", "nm"],
                scales=[1] + list(ds.voxel_size),
            ),
            voxel_offset=[0] + list(ds.roi.offset),
        )
        new_state = copy.deepcopy(self.viewer.state)
        if len(new_state.layers) == 1:
            new_state.layers[layer_name] = neuroglancer.ImageLayer(source=source)
        else:
            # replace name everywhere to preserve state, like what is selected
            new_state_str = json.dumps(new_state.to_json())
            new_state_str = new_state_str.replace(new_state.layers[-1].name, layer_name)
            new_state = ViewerState(json.loads(new_state_str))
            new_state.layers[layer_name].source = source

        self.viewer.set_state(new_state)
        print(self.viewer.state)

    def deprecated_start_neuroglancer(self):
        neuroglancer.set_server_bind_address("0.0.0.0")
        self.viewer = neuroglancer.Viewer()

    def start_neuroglancer(self):
        neuroglancer.set_server_bind_address("0.0.0.0")
        self.viewer = neuroglancer.Viewer()
        with self.viewer.txn() as state:
            state.showSlices = False

            state.layers["raw"] = neuroglancer.ImageLayer(
                source=neuroglancer.LocalVolume(
                    data=self.raw.data,
                    dimensions=neuroglancer.CoordinateSpace(
                        names=["z", "y", "x"],
                        units=["nm", "nm", "nm"],
                        scales=self.raw.voxel_size,
                    ),
                    voxel_offset=self.raw.roi.offset,
                ),
            )
        return IFrame(src=self.viewer, width=1800, height=900)

    def start(self):
        self.array_store = create_array_store()
        self.get_datasets()
        self.new_validation_checker()
        return self.start_neuroglancer()

    def open_from_array_identitifier(self, array_identifier):
        if os.path.exists(array_identifier.container / array_identifier.dataset):
            return open_ds(str(array_identifier.container), array_identifier.dataset)
        else:
            return None

    def get_datasets(self):
        for validation_dataset in self.run.datasplit.validate:
            (
                input_raw_array_identifier,
                input_gt_array_identifier,
            ) = self.array_store.validation_input_arrays(
                self.run.name, validation_dataset.name
            )

            self.raw = self.open_from_array_identitifier(input_raw_array_identifier)
            self.gt = self.open_from_array_identitifier(input_gt_array_identifier)
        print(self.raw)

    def update_best_info(self, iteration, validation_dataset_name):
        prediction_array_identifier = self.array_store.validation_prediction_array(
            self.run.name,
            iteration,
            validation_dataset_name,
        )
        self.prediction = self.open_from_array_identitifier(prediction_array_identifier)
        self.most_recent_iteration = iteration

    def update_neuroglancer(self, iteration):
        self.updated_neuroglancer_layer(
            f"prediction at iteration {iteration}", self.prediction
        )
        return None

    def update_best(self, iteration, validation_dataset_name):
        self.update_best_info(iteration, validation_dataset_name)
        self.update_neuroglancer(iteration)

    def new_validation_checker(self):
        self.process = Thread(target=self.update_with_new_validation_if_possible)
        self.process.daemon = True
        self.process.start()

    def update_with_new_validation_if_possible(self):
        # Here we are assuming that we are checking the directory .../valdiation_config/prediction
        # Ideally we will only have to check for the current best validation
        while True:
            time.sleep(3)
            for validation_dataset in self.run.datasplit.validate:
                most_recent_iteration_previous = self.most_recent_iteration
                prediction_array_identifier = (
                    self.array_store.validation_prediction_array(
                        self.run.name,
                        self.most_recent_iteration,
                        validation_dataset.name,
                    )
                )

                container = prediction_array_identifier.container
                if os.path.exists(container):
                    iteration_dirs = [
                        name
                        for name in os.listdir(container)
                        if os.path.isdir(os.path.join(container, name))
                        and name.isnumeric()
                    ]

                    for iteration_dir in iteration_dirs:
                        if int(iteration_dir) > self.most_recent_iteration:
                            inference_dir = os.path.join(
                                container,
                                iteration_dir,
                                "validation_config",
                                "prediction",
                            )
                            if os.path.exists(inference_dir):
                                # Ignore basic zarr and n5 files
                                inference_dir_contents = [
                                    f
                                    for f in os.listdir(inference_dir)
                                    if not f.startswith(".") and not f.endswith(".json")
                                ]
                                if inference_dir_contents:
                                    # then it should have at least a chunk writtent out, assume it has all of it written out
                                    self.most_recent_iteration = int(iteration_dir)
                    if most_recent_iteration_previous != self.most_recent_iteration:
                        self.update_best(
                            self.most_recent_iteration,
                            validation_dataset.name,
                        )
