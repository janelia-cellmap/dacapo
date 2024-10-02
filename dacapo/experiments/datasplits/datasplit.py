from .datasets import Dataset
import neuroglancer
from abc import ABC
from typing import List, Optional
import json
import itertools


class DataSplit(ABC):
    """
    A class for creating a simple train dataset and no validation dataset. It is derived from `DataSplit` class.
    It is used to split the data into training and validation datasets. The training and validation datasets are
    used to train and validate the model respectively.

    Attributes:
        train : list
            The list containing training datasets. In this class, it contains only one dataset for training.
        validate : list
            The list containing validation datasets. In this class, it is an empty list as no validation dataset is set.
    Methods:
        __init__(self, datasplit_config):
            The constructor for DummyDataSplit class. It initialises a list with training datasets according to the input configuration.
    Notes:
        This class is used to split the data into training and validation datasets.
    """

    train: List[Dataset]
    validate: Optional[List[Dataset]]

    def _neuroglancer(self, embedded=False, bind_address="0.0.0.0", bind_port=0):
        """
        A method to visualize the data in Neuroglancer. It creates a Neuroglancer viewer and adds the layers of the training and validation datasets to it.

        Args:
            embedded : bool
                A boolean flag to indicate if the Neuroglancer viewer is to be embedded in the notebook.
            bind_address : str
                Bind address for Neuroglancer webserver
            bind_port : int
                Bind port for Neuroglancer webserver
        Returns:
            viewer : obj
                The Neuroglancer viewer object.
        Raises:
            Exception
                If the model setup cannot be loaded, an Exception is thrown which is logged and handled by training the model without head matching.
        Examples:
            >>> viewer = datasplit._neuroglancer(embedded=True)
        Notes:
            This function is called by the DataSplit class to visualize the data in Neuroglancer.
            It creates a Neuroglancer viewer and adds the layers of the training and validation datasets to it.
            Neuroglancer is a powerful tool for visualizing large-scale volumetric data.
        """
        neuroglancer.set_server_bind_address(
            bind_address=bind_address, bind_port=bind_port
        )
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

            for k, layer in itertools.chain(
                train_layers.items(), validate_layers.items()
            ):
                s.layers[k] = layer

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
