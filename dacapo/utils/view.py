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
from typing import Optional


def get_viewer(
    arrays: dict,
    width: int = 1500,
    height: int = 600,
    headless: bool = True,
    bind_address: str = "0.0.0.0",
    bind_port: int = 0,
) -> neuroglancer.Viewer | IFrame:
    """
    Creates a neuroglancer viewer to visualize arrays.

    Args:
        arrays (dict): A dictionary containing arrays to be visualized.
        width (int, optional): The width of the viewer window in pixels. Defaults to 1500.
        height (int, optional): The height of the viewer window in pixels. Defaults to 600.
        headless (bool, optional): If True, returns the viewer object. If False, returns an IFrame object embedding the viewer. Defaults to True.
        bind_address (str, optional): Bind address for Neuroglancer webserver.
        bind_port (int, optional): Bind port for Neuroglancer webserver.
    Returns:
        neuroglancer.Viewer | IFrame: The neuroglancer viewer object or an IFrame object embedding the viewer.
    Raises:
        ValueError: If the array is not a numpy array or a neuroglancer.LocalVolume object.
    Examples:
        >>> import numpy as np
        >>> import neuroglancer
        >>> from dacapo.utils.view import get_viewer
        >>> arrays = {
        ...     "raw": {
        ...         "array": np.random.rand(100, 100, 100)
        ...     },
        ...     "seg": {
        ...         "array": np.random.randint(0, 10, (100, 100, 100)),
        ...         "is_seg": True
        ...     }
        ... }
        >>> viewer = get_viewer(arrays)
        >>> viewer
    """

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

    neuroglancer.set_server_bind_address(bind_address=bind_address, bind_port=bind_port)
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
    """
    Add a segmentation layer to the Neuroglancer state.

    Args:
        state (neuroglancer.ViewerState): The Neuroglancer viewer state.
        name (str): The name of the segmentation layer.
        data (ndarray): The segmentation data.
        voxel_size (list): The voxel size in nm.
        meshes (bool, optional): Whether to generate meshes for the segments. Defaults to False.
    Raises:
        ValueError: If the data is not a numpy array.
    Examples:
        >>> import numpy as np
        >>> import neuroglancer
        >>> from dacapo.utils.view import add_seg_layer
        >>> state = neuroglancer.ViewerState()
        >>> data = np.random.randint(0, 10, (100, 100, 100))
        >>> voxel_size = [1, 1, 1]
        >>> add_seg_layer(state, "seg", data, voxel_size)
    """
    if meshes:
        kwargs = {"segments": np.unique(data[data > 0])}
    else:
        kwargs = {}
    state.layers[name] = neuroglancer.SegmentationLayer(
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
    """
    Add a scalar layer to the state.

    Args:
        state (neuroglancer.ViewerState): The viewer state to add the layer to.
        name (str): The name of the layer.
        data (ndarray): The scalar data to be displayed.
        voxel_size (list): The voxel size in nm.
    Raises:
        ValueError: If the data is not a numpy array.
    Examples:
        >>> import numpy as np
        >>> import neuroglancer
        >>> from dacapo.utils.view import add_scalar_layer
        >>> state = neuroglancer.ViewerState()
        >>> data = np.random.rand(100, 100, 100)
        >>> voxel_size = [1, 1, 1]
        >>> add_scalar_layer(state, "raw", data, voxel_size)
    """
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
    """
    Represents the best score achieved during a run.

    Attributes:
        run (Run): The run object associated with the best score.
        score (float): The best score achieved.
        iteration (int): The iteration number at which the best score was achieved.
        parameter (Optional[str]): The parameter associated with the best score.
        validation_parameters: The validation parameters used during the run.
        array_store: The array store object used to store prediction arrays.
        stats_store: The stats store object used to store iteration scores.
        ds: The dataset object associated with the best score.
    Methods:
        get_ds(iteration, validation_dataset): Retrieves the dataset object associated with the best score.
        does_new_best_exist(): Checks if a new best score exists.
    """

    def __init__(self, run: Run):
        """
        Initializes a new instance of the BestScore class.

        Args:
            run (Run): The run object associated with the best score.
        Examples:
            >>> from dacapo.experiments.run import Run
            >>> from dacapo.utils.view import BestScore
            >>> run = Run()
            >>> best_score = BestScore(run)
        """
        self.run: Run = run
        self.score: float = -1
        self.iteration: int = 0
        self.parameter: Optional[str] = None
        self.validation_parameters = run.validation_scores.parameters

        self.array_store = create_array_store()
        self.stats_store = create_stats_store()

    def get_ds(self, iteration, validation_dataset):
        """
        Retrieves the dataset object associated with the best score.

        Args:
            iteration (int): The iteration number.
            validation_dataset: The validation dataset object.
        Raises:
            FileNotFoundError: If the dataset object does not exist.
        Examples:
            >>> from dacapo.experiments.run import Run
            >>> from dacapo.utils.view import BestScore
            >>> run = Run()
            >>> best_score = BestScore(run)
            >>> iteration = 0
            >>> validation_dataset = run.datasplit.validate[0]
            >>> best_score.get_ds(iteration, validation_dataset)
        """
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
        """
        Checks if a new best score exists.

        Returns:
            bool: True if a new best score exists, False otherwise.
        Raises:
            FileNotFoundError: If the dataset object does not exist.
        Examples:
            >>> from dacapo.experiments.run import Run
            >>> from dacapo.utils.view import BestScore
            >>> run = Run()
            >>> best_score = BestScore(run)
            >>> new_best_exists = best_score.does_new_best_exist()
        """
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
    """
    A class for viewing neuroglancer runs.

    Attributes:
        run (Run): The run object.
        best_score (BestScore): The best score object.
        embedded (bool): Whether the viewer is embedded.
        viewer: The neuroglancer viewer.
        raw: The raw dataset.
        gt: The ground truth dataset.
        segmentation: The segmentation dataset.
        most_recent_iteration: The most recent iteration.
        run_thread: The run thread.
        array_store: The array store object.
    Methods:
        updated_neuroglancer_layer(layer_name, ds): Update the neuroglancer layer with the given name and data source.
        deprecated_start_neuroglancer(): Deprecated method to start the neuroglancer viewer.
        start_neuroglancer(): Start the neuroglancer viewer.
        start(): Start the viewer.
        open_from_array_identitifier(array_identifier): Open the array from the given identifier.
        get_datasets(): Get the datasets for validation.
        update_best_info(): Update the best info.
        update_neuroglancer(): Update the neuroglancer viewer.
        update_best_layer(): Update the best layer.
        new_validation_checker(): Start a new validation checker thread.
        update_with_new_validation_if_possible(): Update with new validation if possible.
        stop(): Stop the viewer.
    """

    def __init__(self, run: Run, embedded=False):
        """
        Initialize a View object.

        Args:
            run (Run): The run object.
            embedded (bool, optional): Whether the viewer is embedded. Defaults to False.
        Returns:
            View: The view object.
        Raises:
            FileNotFoundError: If the dataset object does not exist.
        Examples:
            >>> from dacapo.experiments.run import Run
            >>> from dacapo.utils.view import NeuroglancerRunViewer
            >>> run = Run()
            >>> viewer = NeuroglancerRunViewer(run)
        """
        self.run: Run = run
        self.best_score = BestScore(run)
        self.embedded = embedded

    def updated_neuroglancer_layer(self, layer_name, ds):
        """
        Update the neuroglancer layer with the given name and data source.

        Args:
            layer_name (str): The name of the layer.
            ds: The data source.
        Raises:
            FileNotFoundError: If the dataset object does not exist.
        Examples:
            >>> from dacapo.experiments.run import Run
            >>> from dacapo.utils.view import NeuroglancerRunViewer
            >>> run = Run()
            >>> viewer = NeuroglancerRunViewer(run)
            >>> layer_name = "prediction"
            >>> ds = viewer.run.datasplit.validate[0].raw._source_array
            >>> viewer.updated_neuroglancer_layer(layer_name, ds)
        """
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
        """
        Deprecated method to start the neuroglancer viewer.

        Returns:
            IFrame: The embedded viewer.
        Raises:
            FileNotFoundError: If the dataset object does not exist.
        Examples:
            >>> from dacapo.experiments.run import Run
            >>> from dacapo.utils.view import NeuroglancerRunViewer
            >>> run = Run()
            >>> viewer = NeuroglancerRunViewer(run)
            >>> viewer.deprecated_start_neuroglancer()
        """
        neuroglancer.set_server_bind_address("0.0.0.0")
        self.viewer = neuroglancer.Viewer()

    def start_neuroglancer(self, bind_address="0.0.0.0", bind_port=None):
        """
        Start the neuroglancer viewer.

        Args:
            bind_address (str, optional): Bind address for Neuroglancer webserver.
            bind_port (int, optional): Bind port for Neuroglancer webserver.
        Returns:
            IFrame: The embedded viewer.
        Raises:
            FileNotFoundError: If the dataset object does not exist.
        Examples:
            >>> from dacapo.experiments.run import Run
            >>> from dacapo.utils.view import NeuroglancerRunViewer
            >>> run = Run()
            >>> viewer = NeuroglancerRunViewer(run)
            >>> viewer.start_neuroglancer()
        """
        neuroglancer.set_server_bind_address(
            bind_address=bind_address, bind_port=bind_port
        )
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
        """
        Start the viewer.

        Returns:
            IFrame: The embedded viewer.
        Raises:
            FileNotFoundError: If the dataset object does not exist.
        Examples:
            >>> from dacapo.experiments.run import Run
            >>> from dacapo.utils.view import NeuroglancerRunViewer
            >>> run = Run()
            >>> viewer = NeuroglancerRunViewer(run)
            >>> viewer.start()
        """
        self.run_thread = True
        self.array_store = create_array_store()
        self.get_datasets()
        self.new_validation_checker()
        return self.start_neuroglancer()

    def open_from_array_identitifier(self, array_identifier):
        """
        Open the array from the given identifier.

        Args:
            array_identifier: The array identifier.
        Returns:
            The opened dataset or None if it doesn't exist.
        Raises:
            FileNotFoundError: If the dataset object does not exist.
        Examples:
            >>> from dacapo.experiments.run import Run
            >>> from dacapo.utils.view import NeuroglancerRunViewer
            >>> run = Run()
            >>> viewer = NeuroglancerRunViewer(run)
            >>> array_identifier = viewer.run.datasplit.validate[0].raw._source_array
            >>> ds = viewer.open_from_array_identitifier(array_identifier)
        """
        if os.path.exists(array_identifier.container / array_identifier.dataset):
            return open_ds(
                str(array_identifier.container.path), array_identifier.dataset
            )
        else:
            return None

    def get_datasets(self):
        """
        Get the datasets for validation.

        Args:
            run (Run): The run object.
        Returns:
            The raw and ground truth datasets for validation.
        Raises:
            FileNotFoundError: If the dataset object does not exist.
        Examples:
            >>> from dacapo.experiments.run import Run
            >>> from dacapo.utils.view import NeuroglancerRunViewer
            >>> run = Run()
            >>> viewer = NeuroglancerRunViewer(run)
            >>> viewer.get_datasets()
        """
        for validation_dataset in self.run.datasplit.validate:
            raw = validation_dataset.raw._source_array
            gt = validation_dataset.gt._source_array
            self.raw = open_ds(str(raw.file_name), raw.dataset)
            self.gt = open_ds(str(gt.file_name), gt.dataset)

    def update_best_info(self):
        """
        Update the best info.

        Args:
            run (Run): The run object.
        Returns:
            IFrame: The embedded viewer.
        Raises:
            FileNotFoundError: If the dataset object does not exist.
        Examples:
            >>> from dacapo.experiments.run import Run
            >>> from dacapo.utils.view import NeuroglancerRunViewer
            >>> run = Run()
            >>> viewer = NeuroglancerRunViewer(run)
            >>> viewer.update_best_info()
        """
        self.segmentation = self.best_score.ds
        self.most_recent_iteration = self.best_score.iteration

    def update_neuroglancer(self):
        """
        Update the neuroglancer viewer.

        Args:
            run (Run): The run object.
        Raises:
            FileNotFoundError: If the dataset object does not exist.
        Examples:
            >>> from dacapo.experiments.run import Run
            >>> from dacapo.utils.view import NeuroglancerRunViewer
            >>> run = Run()
            >>> viewer = NeuroglancerRunViewer(run)
            >>> viewer.update_neuroglancer()
        """
        self.updated_neuroglancer_layer(
            f"prediction at iteration {self.best_score.iteration}, f1 score {self.best_score.score}",
            self.segmentation,
        )
        return None

    def update_best_layer(self):
        """
        Update the best layer.

        Args:
            run (Run): The run object.
        Returns:
            IFrame: The embedded viewer.
        Raises:
            FileNotFoundError: If the dataset object does not exist.
        Examples:
            >>> from dacapo.experiments.run import Run
            >>> from dacapo.utils.view import NeuroglancerRunViewer
            >>> run = Run()
            >>> viewer = NeuroglancerRunViewer(run)
            >>> viewer.update_best_layer()
        """
        self.update_best_info()
        self.update_neuroglancer()

    def new_validation_checker(self):
        """
        Start a new validation checker thread.

        Args:
            run (Run): The run object.
        Returns:
            IFrame: The embedded viewer.
        Raises:
            FileNotFoundError: If the dataset object does not exist.
        Examples:
            >>> from dacapo.experiments.run import Run
            >>> from dacapo.utils.view import NeuroglancerRunViewer
            >>> run = Run()
            >>> viewer = NeuroglancerRunViewer(run)
            >>> viewer.new_validation_checker()
        """
        self.thread = threading.Thread(
            target=self.update_with_new_validation_if_possible, daemon=True
        )
        self.thread.run_thread = True
        self.thread.start()

    def update_with_new_validation_if_possible(self):
        """
        Update with new validation if possible.

        Args:
            run (Run): The run object.
        Returns:
            IFrame: The embedded viewer.
        Raises:
            FileNotFoundError: If the dataset object does not exist.
        Examples:
            >>> from dacapo.experiments.run import Run
            >>> from dacapo.utils.view import NeuroglancerRunViewer
            >>> run = Run()
            >>> viewer = NeuroglancerRunViewer(run)
            >>> viewer.update_with_new_validation_if_possible()
        """
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
        """
        Stop the viewer.

        Args:
            run (Run): The run object.
        Raises:
            FileNotFoundError: If the dataset object does not exist.
        Returns:
            IFrame: The embedded viewer.
        Examples:
            >>> from dacapo.experiments.run import Run
            >>> from dacapo.utils.view import NeuroglancerRunViewer
            >>> run = Run()
            >>> viewer = NeuroglancerRunViewer(run)
            >>> viewer.stop()
        """
        self.thread.run_thread = False
