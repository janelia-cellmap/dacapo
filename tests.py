import unittest
import numpy as np
from skimage import data
from skimage.filters import gaussian
from scipy.ndimage import label
from funlib.geometry import Coordinate, Roi
from funlib.persistence import prepare_ds
from dacapo.utils.affinities import seg_to_affgraph
from dacapo.store.create_store import create_config_store
from dacapo.experiments.datasplits import DataSplitGenerator, DatasetSpec
from dacapo.experiments.tasks import DistanceTaskConfig, AffinitiesTaskConfig
from dacapo.experiments.architectures import CNNectomeUNetConfig
from dacapo.experiments.trainers import GunpowderTrainerConfig
from dacapo.experiments.run import RunConfig
from dacapo.train import train_run
import zarr

class TestDacapoPipeline(unittest.TestCase):
    def setUp(self):
        """Set up shared resources for tests."""
        self.voxel_size = Coordinate(290, 260, 260)
        self.cell_data = (data.cells3d().transpose((1, 0, 2, 3)) / 256).astype(np.uint8)
        self.roi = Roi((0, 0, 0), self.cell_data.shape[1:]) * self.voxel_size
        self.config_store = create_config_store()

    def test_prepare_raw_data(self):
        """Test preparing the raw data."""
        cell_array = prepare_ds(
            "test_cells3d.zarr",
            "raw",
            self.roi,
            voxel_size=self.voxel_size,
            dtype=np.uint8,
            num_channels=None,
        )
        cell_array[cell_array.roi] = self.cell_data[1]
        self.assertEqual(cell_array.shape, self.cell_data.shape[1:])
        self.assertEqual(cell_array.dtype, np.uint8)

    def test_generate_mask(self):
        """Test generating the mask data."""
        mask_array = prepare_ds(
            "test_cells3d.zarr",
            "mask",
            self.roi,
            voxel_size=self.voxel_size,
            dtype=np.uint8,
        )
        cell_mask = np.clip(gaussian(self.cell_data[1] / 255.0, sigma=1), 0, 255) * 255 > 30
        not_membrane_mask = np.clip(gaussian(self.cell_data[0] / 255.0, sigma=1), 0, 255) * 255 < 10
        mask_array[mask_array.roi] = cell_mask * not_membrane_mask
        self.assertTrue(np.any(mask_array.to_ndarray(mask_array.roi)))

    def test_connected_components(self):
        """Test generating connected components."""
        labels_array = prepare_ds(
            "test_cells3d.zarr",
            "labels",
            self.roi,
            voxel_size=self.voxel_size,
            dtype=np.uint8,
        )
        mask_array = np.zeros(self.cell_data.shape[1:], dtype=np.uint8)
        labels_array[labels_array.roi] = label(mask_array)[0]
        self.assertTrue(np.any(labels_array.to_ndarray(labels_array.roi)))

    def test_generate_affinities(self):
        """Test generating affinities."""
        labels_array = np.random.randint(0, 5, size=self.cell_data.shape[1:], dtype=np.uint8)
        affs_array = prepare_ds(
            "test_cells3d.zarr",
            "affs",
            self.roi,
            voxel_size=self.voxel_size,
            num_channels=3,
            dtype=np.uint8,
        )
        affs = seg_to_affgraph(
            labels_array,
            neighborhood=[Coordinate(1, 0, 0), Coordinate(0, 1, 0), Coordinate(0, 0, 1)],
        )
        affs_array[affs_array.roi] = affs * 255
        self.assertEqual(affs_array.shape, (3, *self.cell_data.shape[1:]))

    def test_configurations(self):
        """Test storing configurations."""
        dataspecs = [
            DatasetSpec(
                dataset_type="train",
                raw_container="test_cells3d.zarr",
                raw_dataset="raw",
                gt_container="test_cells3d.zarr",
                gt_dataset="mask",
            ),
        ]
        datasplit_config = DataSplitGenerator(
            name="test_data",
            datasets=dataspecs,
            input_resolution=self.voxel_size,
            output_resolution=self.voxel_size,
            targets=["cell"],
        ).compute()
        self.config_store.store_datasplit_config(datasplit_config)
        retrieved = self.config_store.retrieve_datasplit_config("test_data")
        self.assertEqual(retrieved.name, "test_data")

    def test_training(self):
        """Test training functionality."""
        datasplit_config = self.config_store.retrieve_datasplit_config("test_data")
        affs_task_config = AffinitiesTaskConfig(name="test_affs", neighborhood=[(0, 1, 0), (0, 0, 1)])
        self.config_store.store_task_config(affs_task_config)
        architecture_config = CNNectomeUNetConfig(name="test_unet", input_shape=(2, 64, 64))
        trainer_config = GunpowderTrainerConfig(name="test_trainer", batch_size=2)
        run_config = RunConfig(
            name="test_run",
            datasplit_config=datasplit_config,
            task_config=affs_task_config,
            architecture_config=architecture_config,
            trainer_config=trainer_config,
            num_iterations=10,
        )
        self.config_store.store_run_config(run_config)
        run = Run(self.config_store.retrieve_run_config("test_run"))
        train_run(run)
        stats_store = create_stats_store()
        training_stats = stats_store.retrieve_training_stats("test_run")
        self.assertIsNotNone(training_stats)

if __name__ == "__main__":
    unittest.main()
