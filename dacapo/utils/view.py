import neuroglancer
from IPython.display import IFrame
import numpy as np
from funlib.persistence import open_ds
import threading
import neuroglancer
from neuroglancer.viewer_state import ViewerState
import os
from dacapo.experiments.run import Run
from dacapo.store.create_store import create_array_store, create_stats_store
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


class BestScore:
    def __init__(self, run: Run):
        self.run: Run = run
        self.score: float = -1
        self.iteration: int = 0
        self.parameter: str = None
        self.validation_parameters = run.validation_scores.parameters

        self.array_store = create_array_store()
        self.stats_store = create_stats_store()

    def get_ds(self, iteration, validation_dataset):
        prediction_array_identifier = self.array_store.validation_prediction_array(
            self.run.name, iteration, validation_dataset.name
        )
        container = str(prediction_array_identifier.container)
        dataset = str(
            os.path.join(
                str(iteration), validation_dataset.name, "output", str(self.parameter)
            )
        )
        self.ds = open_ds(container, dataset)

    def does_new_best_exist(self):
        new_best_exists = False
        self.validation_scores = self.stats_store.retrieve_validation_iteration_scores(
            self.run.name
        )

        for validation_idx, validation_dataset in enumerate(
            self.run.datasplit.validate
        ):
            for iteration_scores in self.validation_scores:
                iteration = iteration_scores.iteration
                for parameter_idx, parameter in enumerate(self.validation_parameters):
                    # hardcoded for f1_score
                    current_score = iteration_scores.scores[validation_idx][
                        parameter_idx
                    ][20]
                    if current_score > self.score:
                        self.iteration = iteration
                        self.score = current_score
                        self.parameter = parameter
                        self.get_ds(iteration, validation_dataset)
                        new_best_exists = True
        return new_best_exists


class NeuroglancerRunViewer:
    def __init__(self, run: Run, embedded=False):
        self.run: Run = run
        self.best_score = BestScore(run)
        self.embedded = embedded

    def updated_neuroglancer_layer(self, layer_name, ds):
        source = neuroglancer.LocalVolume(
            data=ds.data,
            dimensions=neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                units=["nm", "nm", "nm"],
                scales=list(ds.voxel_size),
            ),
            voxel_offset=list(ds.roi.offset),
        )
        new_state = copy.deepcopy(self.viewer.state)
        if len(new_state.layers) == 1:
            new_state.layers[layer_name] = neuroglancer.SegmentationLayer(source=source)
        else:
            # replace name everywhere to preserve state, like what is selected
            new_state_str = json.dumps(new_state.to_json())
            new_state_str = new_state_str.replace(new_state.layers[-1].name, layer_name)
            new_state = ViewerState(json.loads(new_state_str))
            new_state.layers[layer_name].source = source

        self.viewer.set_state(new_state)

    def deprecated_start_neuroglancer(self):
        neuroglancer.set_server_bind_address("0.0.0.0")
        self.viewer = neuroglancer.Viewer()

    def start_neuroglancer(self):
        neuroglancer.set_server_bind_address("0.0.0.0")
        self.viewer = neuroglancer.Viewer()
        print(f"Neuroglancer viewer: {self.viewer}")
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
        if self.embedded:
            return IFrame(src=self.viewer, width=1800, height=900)

    def start(self):
        self.run_thread = True
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
            raw = validation_dataset.raw._source_array
            gt = validation_dataset.gt._source_array
            self.raw = open_ds(str(raw.file_name), raw.dataset)
            self.gt = open_ds(str(gt.file_name), gt.dataset)

    def update_best_info(self):
        self.segmentation = self.best_score.ds
        self.most_recent_iteration = self.best_score.iteration

    def update_neuroglancer(self):
        self.updated_neuroglancer_layer(
            f"prediction at iteration {self.best_score.iteration}, f1 score {self.best_score.score}",
            self.segmentation,
        )
        return None

    def update_best_layer(self):
        self.update_best_info()
        self.update_neuroglancer()

    def new_validation_checker(self):
        self.thread = threading.Thread(
            target=self.update_with_new_validation_if_possible, daemon=True
        )
        self.thread.run_thread = True
        self.thread.start()

    def update_with_new_validation_if_possible(self):
        thread = threading.currentThread()
        # Here we are assuming that we are checking the directory .../valdiation_config/prediction
        # Ideally we will only have to check for the current best validation
        while getattr(thread, "run_thread", True):
            time.sleep(10)
            new_best_exists = self.best_score.does_new_best_exist()
            if new_best_exists:
                print(
                    f"New best f1 score of {self.best_score.score} at iteration {self.best_score.iteration} and parameter {self.best_score.parameter}"
                )
                self.update_best_layer()

    def stop(self):
        self.thread.run_thread = False
