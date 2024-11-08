from dacapo.experiments.tasks import TaskConfig
from dacapo.experiments.datasplits.datasets.arrays import ArrayConfig
from upath import UPath as Path
from typing import List, Union, Optional, Sequence
from enum import Enum, EnumMeta
from funlib.geometry import Coordinate

import zarr
from zarr.n5 import N5FSStore
import numpy as np
from dacapo.experiments.datasplits.datasets.arrays import (
    ResampledArrayConfig,
    BinarizeArrayConfig,
    IntensitiesArrayConfig,
    ConcatArrayConfig,
    LogicalOrArrayConfig,
    ConstantArrayConfig,
    CropArrayConfig,
    ZarrArrayConfig,
)
from dacapo.experiments.datasplits import TrainValidateDataSplitConfig
from dacapo.experiments.datasplits.datasets import RawGTDatasetConfig
import logging


logger = logging.getLogger(__name__)


def is_zarr_group(file_name: Path, dataset: str):
    """
    Check if the dataset is a Zarr group. If the dataset is a Zarr group, it will return True, otherwise False.

    Args:
        file_name : str
            The name of the file.
        dataset : str
            The name of the dataset.
    Returns:
        bool : True if the dataset is a Zarr group, otherwise False.
    Raises:
        FileNotFoundError
            If the file does not exist, a FileNotFoundError is raised.
    Examples:
        >>> is_zarr_group(file_name, dataset)
    Notes:
        This function is used to check if the dataset is a Zarr group.
    """
    if file_name.suffix == ".n5":
        zarr_file = zarr.open(N5FSStore(str(file_name)), mode="r")
    else:
        zarr_file = zarr.open(str(file_name), mode="r")
    return isinstance(zarr_file[dataset], zarr.hierarchy.Group)


def resize_if_needed(
    array_config: ZarrArrayConfig, target_resolution: Coordinate, extra_str=""
):
    """
    Resize the array if needed. If the array needs to be resized, it will return the resized array, otherwise it will return the original array.

    Args:
        array_config : obj
            The configuration of the array.
        target_resolution : obj
            The target resolution.
        extra_str : str
            An extra string.
    Returns:
        obj : The resized array if needed, otherwise the original array.
    Raises:
        FileNotFoundError
            If the file does not exist, a FileNotFoundError is raised.
    Examples:
        >>> resize_if_needed(array_config, target_resolution, extra_str)
    Notes:
        This function is used to resize the array if needed.
    """
    zarr_array = array_config.array()
    raw_voxel_size = zarr_array.voxel_size

    raw_upsample = raw_voxel_size / target_resolution
    raw_downsample = target_resolution / raw_voxel_size
    assert len(target_resolution) == zarr_array.dims, (
        f"Target resolution {target_resolution} and raw voxel size {raw_voxel_size} "
        f"have different dimensions {zarr_array.dims}"
    )
    if any([u > 1 or d > 1 for u, d in zip(raw_upsample, raw_downsample)]):
        print(
            f"dataset {array_config} needs resampling to {target_resolution}, upsample: {raw_upsample}, downsample: {raw_downsample}"
        )
        return ResampledArrayConfig(
            name=f"{extra_str}_{array_config.name}_{array_config.dataset}_resampled",
            source_array_config=array_config,
            upsample=raw_upsample,
            downsample=raw_downsample,
            interp_order=False,
        )
    else:
        # print(f"dataset {array_config.dataset} does not need resampling found {raw_voxel_size}=={target_resolution}")
        return array_config


def limit_validation_crop_size(gt_config, mask_config, max_size):
    gt_array = gt_config.array()
    voxel_shape = gt_array.roi.shape / gt_array.voxel_size
    crop = False
    while np.prod(voxel_shape) > max_size:
        crop = True
        max_idx = np.argmax(voxel_shape)
        voxel_shape = Coordinate(
            s if i != max_idx else s // 2 for i, s in enumerate(voxel_shape)
        )
    if crop:
        crop_roi_shape = voxel_shape * gt_array.voxel_size
        context = (gt_array.roi.shape - crop_roi_shape) / 2
        crop_roi = gt_array.roi.grow(-context, -context)
        crop_roi = crop_roi.snap_to_grid(gt_array.voxel_size, mode="shrink")

        logger.debug(
            f"Cropped {gt_config.name}: original roi: {gt_array.roi}, new_roi: {crop_roi}"
        )

        gt_config = CropArrayConfig(
            name=gt_config.name + "_cropped",
            source_array_config=gt_config,
            roi=crop_roi,
        )
        mask_config = CropArrayConfig(
            name=mask_config.name + "_cropped",
            source_array_config=gt_config,
            roi=crop_roi,
        )
    return gt_config, mask_config


def get_right_resolution_array_config(
    container: Path, dataset, target_resolution, extra_str=""
):
    """
    Get the right resolution array configuration. It will return the right resolution array configuration.

    Args:
        container : obj
            The container.
        dataset : str
            The dataset.
        target_resolution : obj
            The target resolution.
        extra_str : str
            An extra string.
    Returns:
        obj : The right resolution array configuration.
    Raises:
        FileNotFoundError
            If the file does not exist, a FileNotFoundError is raised.
    Examples:
        >>> get_right_resolution_array_config(container, dataset, target_resolution, extra_str)
    Notes:
        This function is used to get the right resolution array configuration.
    """
    level = 0
    current_dataset_path = Path(dataset, f"s{level}")
    if not (container / current_dataset_path).exists():
        raise FileNotFoundError(
            f"Path {container} is a Zarr Group and /s0 does not exist."
        )

    zarr_config = ZarrArrayConfig(
        name=f"{extra_str}_{container.stem}_{dataset}_uint8",
        file_name=container,
        dataset=str(current_dataset_path),
        snap_to_grid=target_resolution,
        mode="r",
    )
    zarr_array = zarr_config.array()
    while (
        all([z < t for (z, t) in zip(zarr_array.voxel_size, target_resolution)])
        and Path(container, Path(dataset, f"s{level+1}")).exists()
    ):
        level += 1
        zarr_config = ZarrArrayConfig(
            name=f"{extra_str}_{container.stem}_{dataset}_s{level}_uint8",
            file_name=container,
            dataset=str(Path(dataset, f"s{level}")),
            snap_to_grid=target_resolution,
            mode="r",
        )

        zarr_array = zarr_config.array()
    return resize_if_needed(zarr_config, target_resolution, extra_str)


class CustomEnumMeta(EnumMeta):
    """
    Custom Enum Meta class to raise KeyError when an invalid option is passed.

    Attributes:
        _member_names_ : list
            The list of member names.
    Methods:
        __getitem__(self, item)
            A method to get the item.
    Notes:
        This class is used to raise KeyError when an invalid option is passed.
    """

    def __getitem__(self, item):
        """
        Get the item.

        Args:
            item : obj
                The item.
        Returns:
            obj : The item.
        Raises:
            KeyError
            If the item is not a valid option, a KeyError is raised.
        Examples:
            >>> __getitem__(item)
        Notes:
            This function is used to get the item.
        """
        if item not in self._member_names_:
            raise KeyError(
                f"{item} is not a valid option of {self.__name__}, the valid options are {self._member_names_}"
            )
        return super().__getitem__(item)


class CustomEnum(Enum, metaclass=CustomEnumMeta):
    """
    A custom Enum class to raise KeyError when an invalid option is passed.

    Attributes:
        __str__ : str
            The string representation of the class.
    Methods:
        __str__(self)
            A method to get the string representation of the class.
    Notes:
        This class is used to raise KeyError when an invalid option is passed.
    """

    def __str__(self) -> str:
        """
        Get the string representation of the class.

        Args:
            self : obj
                The object.
        Returns:
            str : The string representation of the class.
        Raises:
            KeyError
            If the item is not a valid option, a KeyError is raised.
        Examples:
            >>> __str__()
        Notes:
            This function is used to get the string representation of the class.
        """
        return self.name


class DatasetType(CustomEnum):
    """
    An Enum class to represent the dataset type. It is derived from `CustomEnum` class.

    Attributes:
        val : int
            The value of the dataset type.
        train : int
            The training dataset type.
    Methods:
        __str__(self)
            A method to get the string representation of the class.
    Notes:
        This class is used to represent the dataset type.
    """

    val = 1
    train = 2


class SegmentationType(CustomEnum):
    """
    An Enum class to represent the segmentation type. It is derived from `CustomEnum` class.

    Attributes:
        semantic : int
            The semantic segmentation type.
        instance : int
            The instance segmentation type.
    Methods:
        __str__(self)
            A method to get the string representation of the class.
    Notes:
        This class is used to represent the segmentation type.
    """

    semantic = 1
    instance = 2


class DatasetSpec:
    """
    A class for dataset specification. It is used to specify the dataset.

    Attributes:
        dataset_type : obj
            The dataset type.
        raw_container : obj
            The raw container.
        raw_dataset : str
            The raw dataset.
        gt_container : obj
            The ground truth container.
        gt_dataset : str
            The ground truth dataset.
    Methods:
        __init__(dataset_type, raw_container, raw_dataset, gt_container, gt_dataset)
            Initializes the DatasetSpec class with the specified dataset type, raw container, raw dataset, ground truth container, and ground truth dataset.
        __str__(self)
            A method to get the string representation of the class.
    Notes:
        This class is used to specify the dataset.
    """

    def __init__(
        self,
        dataset_type: Union[str, DatasetType],
        raw_container: Union[str, Path],
        raw_dataset: str,
        gt_container: Union[str, Path],
        gt_dataset: str,
    ):
        """
        Initializes the DatasetSpec class with the specified dataset type, raw container, raw dataset, ground truth container, and ground truth dataset.

        Args:
            dataset_type : obj
                The dataset type.
            raw_container : obj
                The raw container.
            raw_dataset : str
                The raw dataset.
            gt_container : obj
                The ground truth container.
            gt_dataset : str
                The ground truth dataset.
        Raises:
            KeyError
            If the item is not a valid option, a KeyError is raised.
        Methods:
            __init__(dataset_type, raw_container, raw_dataset, gt_container, gt_dataset)
        Notes:
            This function is used to initialize the DatasetSpec class with the specified dataset type, raw container, raw dataset, ground truth container, and ground truth dataset.
        """
        if isinstance(dataset_type, str):
            dataset_type = DatasetType[dataset_type.lower()]

        if isinstance(raw_container, str):
            raw_container = Path(raw_container)

        if isinstance(gt_container, str):
            gt_container = Path(gt_container)

        self.dataset_type = dataset_type
        self.raw_container = raw_container
        self.raw_dataset = raw_dataset
        self.gt_container = gt_container
        self.gt_dataset = gt_dataset

    def __str__(self) -> str:
        """
        Get the string representation of the class.

        Args:
            self : obj
                The object.
        Returns:
            str : The string representation of the class.
        Raises:
            KeyError
            If the item is not a valid option, a KeyError is raised.
        Examples:
            >>> __str__()
        Notes:
            This function is used to get the string representation of the class.
        """
        return f"{self.raw_container.stem}_{self.gt_dataset}"


def generate_dataspec_from_csv(csv_path: Path):
    """
    Generate the dataset specification from the CSV file. It will return the dataset specification.

    Args:
        csv_path : obj
            The CSV file path.
    Returns:
        list : The dataset specification.
    Raises:
        FileNotFoundError
            If the file does not exist, a FileNotFoundError is raised.
    Examples:
        >>> generate_dataspec_from_csv(csv_path)
    Notes:
        This function is used to generate the dataset specification from the CSV file.
    """
    datasets = []
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file {csv_path} does not exist.")
    with open(csv_path, "r") as f:
        for line in f:
            (
                dataset_type,
                raw_container,
                raw_dataset,
                gt_container,
                gt_dataset,
            ) = line.strip().split(",")
            datasets.append(
                DatasetSpec(
                    dataset_type=DatasetType[dataset_type.lower()],
                    raw_container=Path(raw_container),
                    raw_dataset=raw_dataset,
                    gt_container=Path(gt_container),
                    gt_dataset=gt_dataset,
                )
            )

    return datasets


class DataSplitGenerator:
    """Generates DataSplitConfig for a given task config and datasets.

    Class names in gt_dataset should be within [] e.g. [mito&peroxisome&er] for
    multiple classes or [mito] for one class.

    Currently only supports:
     - semantic segmentation.
     Supports:
        - 2D and 3D datasets.
        - Zarr, N5 and OME-Zarr datasets.
        - Multi class targets.
        - Different resolutions for raw and ground truth datasets.
        - Different resolutions for training and validation datasets.

    Attributes:
        name : str
            The name of the data split generator.
        datasets : list
            The list of dataset specifications.
        input_resolution : obj
            The input resolution.
        output_resolution : obj
            The output resolution.
        targets : list
            The list of targets.
        segmentation_type : obj
            The segmentation type.
        max_gt_downsample : int
            The maximum ground truth downsample.
        max_gt_upsample : int
            The maximum ground truth upsample.
        max_raw_training_downsample : int
            The maximum raw training downsample.
        max_raw_training_upsample : int
            The maximum raw training upsample.
        max_raw_validation_downsample : int
            The maximum raw validation downsample.
        max_raw_validation_upsample : int
            The maximum raw validation upsample.
        min_training_volume_size : int
            The minimum training volume size.
        raw_min : int
            The minimum raw value.
        raw_max : int
            The maximum raw value.
        classes_separator_character : str
            The classes separator character.
        max_validation_volume_size : int
            The maximum validation volume size. Default is None. If None, the validation volume size is not limited.
            else, the validation volume size is limited to the specified value.
            e.g. 600**3 for 600^3 voxels = 216_000_000 voxels.
    Methods:
        __init__(name, datasets, input_resolution, output_resolution, targets, segmentation_type, max_gt_downsample, max_gt_upsample, max_raw_training_downsample, max_raw_training_upsample, max_raw_validation_downsample, max_raw_validation_upsample, min_training_volume_size, raw_min, raw_max, classes_separator_character)
            Initializes the DataSplitGenerator class with the specified name, datasets, input resolution, output resolution, targets, segmentation type, maximum ground truth downsample, maximum ground truth upsample, maximum raw training downsample, maximum raw training upsample, maximum raw validation downsample, maximum raw validation upsample, minimum training volume size, minimum raw value, maximum raw value, and classes separator character.
        __str__(self)
            A method to get the string representation of the class.
        class_name(self)
            A method to get the class name.
        check_class_name(self, class_name)
            A method to check the class name.
        compute(self)
            A method to compute the data split.
        __generate_semantic_seg_datasplit(self)
            A method to generate the semantic segmentation data split.
        __generate_semantic_seg_dataset_crop(self, dataset)
            A method to generate the semantic segmentation dataset crop.
        generate_csv(datasets, csv_path)
            A method to generate the CSV file.
        generate_from_csv(csv_path, input_resolution, output_resolution, name, **kwargs)
            A method to generate the data split from the CSV file.
    Notes:
        - This class is used to generate the DataSplitConfig for a given task config and datasets.
        - Class names in gt_dataset shoulb be within [] e.g. [mito&peroxisome&er] for mutiple classes or [mito] for one class
    """

    def __init__(
        self,
        name: str,
        datasets: List[DatasetSpec],
        input_resolution: Union[Sequence[int], Coordinate],
        output_resolution: Union[Sequence[int], Coordinate],
        targets: Optional[List[str]] = None,
        segmentation_type: Union[str, SegmentationType] = "semantic",
        max_gt_downsample=32,
        max_gt_upsample=4,
        max_raw_training_downsample=16,
        max_raw_training_upsample=2,
        max_raw_validation_downsample=8,
        max_raw_validation_upsample=2,
        min_training_volume_size=8_000,  # 20**3
        raw_min=0,
        raw_max=255,
        classes_separator_character="&",
        use_negative_class=False,
        max_validation_volume_size=None,
        binarize_gt=False,
    ):
        """
        Initializes the DataSplitGenerator class with the specified:
        - name
        - datasets
        - input resolution
        - output resolution
        - targets
        - segmentation type
        - maximum ground truth downsample
        - maximum ground truth upsample
        - maximum raw training downsample
        - maximum raw training upsample
        - maximum raw validation downsample
        - maximum raw validation upsample
        - minimum training volume size
        - minimum raw value
        - maximum raw value
        - classes separator character
        - use negative class
        - binarize ground truth

        Args:
            name : str
                The name of the data split generator.
            datasets : list
                The list of dataset specifications.
            input_resolution : obj
                The input resolution.
            output_resolution : obj
                The output resolution.
            targets : list
                The list of targets.
            segmentation_type : obj
                The segmentation type.
            max_gt_downsample : int
                The maximum ground truth downsample.
            max_gt_upsample : int
                The maximum ground truth upsample.
            max_raw_training_downsample : int
                The maximum raw training downsample.
            max_raw_training_upsample : int
                The maximum raw training upsample.
            max_raw_validation_downsample : int
                The maximum raw validation downsample.
            max_raw_validation_upsample : int
                The maximum raw validation upsample.
            min_training_volume_size : int
                The minimum training volume size.
            raw_min : int
                The minimum raw value.
            raw_max : int
                The maximum raw value.
            classes_separator_character : str
                The classes separator character.
            use_negative_class : bool
                Whether to use negative classes.
            binarize_gt : bool
                Whether to binarize the ground truth as part of preprocessing. Use this if you are doing semantic segmentation on instance labels (where each object has a unique ID).
        Returns:
            obj : The DataSplitGenerator class.
        Raises:
            ValueError
            If the class name is already set, a ValueError is raised.
        Examples:
            >>> DataSplitGenerator(name, datasets, input_resolution, output_resolution, targets, segmentation_type, max_gt_downsample, max_gt_upsample, max_raw_training_downsample, max_raw_training_upsample, max_raw_validation_downsample, max_raw_validation_upsample, min_training_volume_size, raw_min, raw_max, classes_separator_character)
        Notes:
            This function is used to initialize the DataSplitGenerator class with the specified name, datasets, input resolution, output resolution, targets, segmentation type, maximum ground truth downsample, maximum ground truth upsample, maximum raw training downsample, maximum raw training upsample, maximum raw validation downsample, maximum raw validation upsample, minimum training volume size, minimum raw value, maximum raw value, and classes separator character.

        """
        if not isinstance(input_resolution, Coordinate):
            input_resolution = Coordinate(input_resolution)
        if not isinstance(output_resolution, Coordinate):
            output_resolution = Coordinate(output_resolution)
        if isinstance(segmentation_type, str):
            segmentation_type = SegmentationType[segmentation_type.lower()]

        self.name = name
        self.datasets = datasets
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.targets = targets
        self._class_name = None
        self.segmentation_type = segmentation_type
        self.max_gt_downsample = max_gt_downsample
        self.max_gt_upsample = max_gt_upsample
        self.max_raw_training_downsample = max_raw_training_downsample
        self.max_raw_training_upsample = max_raw_training_upsample
        self.max_raw_validation_downsample = max_raw_validation_downsample
        self.max_raw_validation_upsample = max_raw_validation_upsample
        self.min_training_volume_size = min_training_volume_size
        self.raw_min = raw_min
        self.raw_max = raw_max
        self.classes_separator_character = classes_separator_character
        self.use_negative_class = use_negative_class
        self.max_validation_volume_size = max_validation_volume_size
        self.binarize_gt = binarize_gt
        if use_negative_class:
            if targets is None:
                raise ValueError(
                    "use_negative_class=True requires targets to be specified."
                )

    def __str__(self) -> str:
        """
        Get the string representation of the class.

        Args:
            self : obj
                The object.
        Returns:
            str : The string representation of the class.
        Raises:
            ValueError
            If the class name is already set, a ValueError is raised.
        Examples:
            >>> __str__()
        Notes:
            This function is used to get the string representation of the class.
        """
        return f"DataSplitGenerator:{self.name}_{self.segmentation_type}_{self.class_name}_{self.output_resolution[0]}nm"

    @property
    def class_name(self):
        """
        Get the class name.

        Args:
            self : obj
                The object.
        Returns:
            obj : The class name.
        Raises:
            ValueError
            If the class name is already set, a ValueError is raised.
        Examples:
            >>> class_name
        Notes:
            This function is used to get the class name.
        """
        if self._class_name is None:
            if self.targets is None:
                logger.warning("Both targets and class name are None.")
                return None
            self._class_name = self.targets
        return self._class_name

    # Goal is to force class_name to be set only once, so we have the same classes for all datasets
    @class_name.setter
    def class_name(self, class_name):
        """
        Set the class name.

        Args:
            self : obj
                The object.
            class_name : obj
                The class name.
        Returns:
            obj : The class name.
        Raises:
            ValueError
            If the class name is already set, a ValueError is raised.
        Examples:
            >>> class_name
        Notes:
            This function is used to set the class name.
        """
        if self._class_name is not None:
            raise ValueError(
                f"Class name already set. Current class name is {self.class_name} and new class name is {class_name}"
            )
        self._class_name = class_name

    def check_class_name(self, class_name):
        """
        Check the class name.

        Args:
            self : obj
                The object.
            class_name : obj
                The class name.
        Returns:
            obj : The class name.
        Raises:
            ValueError
            If the class name is already set, a ValueError is raised.
        Examples:
            >>> check_class_name(class_name)
        Notes:
            This function is used to check the class name.

        """
        datasets, classes = format_class_name(
            class_name, self.classes_separator_character, self.targets
        )
        if self.class_name is None:
            self.class_name = classes
            if self.targets is None:
                logger.warning(
                    f" No targets specified, using all classes in the dataset as target {classes}."
                )
        elif self.class_name != classes:
            raise ValueError(
                f"Datasets are having different classes names:  {classes} does not match {self.class_name}"
            )
        return datasets, classes

    def compute(self):
        """
        Compute the data split.

        Args:
            self : obj
                The object.
        Returns:
            obj : The data split.
        Raises:
            NotImplementedError
            If the segmentation type is not implemented, a NotImplementedError is raised.
        Examples:
            >>> compute()
        Notes:
            This function is used to compute the data split.
        """
        if self.segmentation_type == SegmentationType.semantic:
            return self.__generate_semantic_seg_datasplit()
        else:
            raise NotImplementedError(
                f"{self.segmentation_type} segmentation not implemented yet!"
            )

    def __generate_semantic_seg_datasplit(self):
        """
        Generate the semantic segmentation data split.

        Args:
            self : obj
                The object.
        Returns:
            obj : The data split.
        Raises:
            FileNotFoundError
            If the file does not exist, a FileNotFoundError is raised.
        Examples:
            >>> __generate_semantic_seg_datasplit()
        Notes:
            This function is used to generate the semantic segmentation data split.

        """
        train_dataset_configs = []
        validation_dataset_configs = []
        for dataset in self.datasets:
            (
                raw_config,
                gt_config,
                mask_config,
            ) = self.__generate_semantic_seg_dataset_crop(dataset)
            if type(self.class_name) == list:
                classes = self.classes_separator_character.join(self.class_name)
            else:
                classes = self.class_name
            if dataset.dataset_type == DatasetType.train:
                train_dataset_configs.append(
                    RawGTDatasetConfig(
                        name=f"{dataset}_{gt_config.name}_{classes}_{self.output_resolution[0]}nm",
                        raw_config=raw_config,
                        gt_config=gt_config,
                        mask_config=mask_config,
                    )
                )
            else:
                if self.max_validation_volume_size is not None:
                    gt_config, mask_config = limit_validation_crop_size(
                        gt_config, mask_config, self.max_validation_volume_size
                    )
                validation_dataset_configs.append(
                    RawGTDatasetConfig(
                        name=f"{dataset}_{gt_config.name}_{classes}_{self.output_resolution[0]}nm",
                        raw_config=raw_config,
                        gt_config=gt_config,
                        mask_config=mask_config,
                    )
                )

        return TrainValidateDataSplitConfig(
            name=f"{self.name}_{self.segmentation_type}_{classes}_{self.output_resolution[0]}nm",
            train_configs=train_dataset_configs,
            validate_configs=validation_dataset_configs,
        )

    def __generate_semantic_seg_dataset_crop(self, dataset: DatasetSpec):
        """
        Generate the semantic segmentation dataset crop.

        Args:
            self : obj
                The object.
            dataset : obj
                The dataset.
        Returns:
            obj : The dataset crop.
        Raises:
            FileNotFoundError
            If the file does not exist, a FileNotFoundError is raised.
        Examples:
            >>> __generate_semantic_seg_dataset_crop(dataset)
        Notes:
            This function is used to generate the semantic segmentation dataset crop.
        """
        raw_container = dataset.raw_container
        raw_dataset = dataset.raw_dataset
        gt_path = dataset.gt_container
        gt_dataset = dataset.gt_dataset

        if not (raw_container / raw_dataset).exists():
            raise FileNotFoundError(
                f"Raw path {raw_container/raw_dataset} does not exist."
            )

        # print(
        #     f"Processing raw_container:{raw_container} raw_dataset:{raw_dataset} gt_path:{gt_path} gt_dataset:{gt_dataset}"
        # )

        if is_zarr_group(raw_container, raw_dataset):
            raw_config = get_right_resolution_array_config(
                raw_container, raw_dataset, self.input_resolution, "raw"
            )
        else:
            raw_config = resize_if_needed(
                ZarrArrayConfig(
                    name=f"raw_{raw_container.stem}_uint8",
                    file_name=raw_container,
                    dataset=raw_dataset,
                    mode="r",
                ),
                self.input_resolution,
                "raw",
            )
        raw_config = IntensitiesArrayConfig(
            name=f"raw_{raw_container.stem}_uint8",
            source_array_config=raw_config,
            min=self.raw_min,
            max=self.raw_max,
        )
        organelle_arrays = {}
        # classes_datasets, classes = self.check_class_name(gt_dataset)
        classes_datasets, classes = format_class_name(
            gt_dataset, self.classes_separator_character, self.targets
        )
        for current_class_dataset, current_class_name in zip(classes_datasets, classes):
            if not (gt_path / current_class_dataset).exists():
                raise FileNotFoundError(
                    f"GT path {gt_path/current_class_dataset} does not exist."
                )
            if is_zarr_group(gt_path, current_class_dataset):
                gt_config = get_right_resolution_array_config(
                    gt_path, current_class_dataset, self.output_resolution, "gt"
                )
            else:
                gt_config = resize_if_needed(
                    ZarrArrayConfig(
                        name=f"gt_{gt_path.stem}_{current_class_dataset}_uint8",
                        file_name=gt_path,
                        dataset=current_class_dataset,
                        mode="r",
                    ),
                    self.output_resolution,
                    "gt",
                )
            if self.binarize_gt:
                gt_config = BinarizeArrayConfig(
                    f"{dataset}_{current_class_name}_{self.output_resolution[0]}nm_binarized",
                    source_array_config=gt_config,
                    groupings=[(current_class_name, [])],
                )
            organelle_arrays[current_class_name] = gt_config

        if self.targets is None:
            targets_str = "_".join(classes)
            current_targets = classes
        else:
            current_targets = self.targets
            targets_str = "_".join(self.targets)

        target_images = dict[str, ArrayConfig]()
        target_masks = dict[str, ArrayConfig]()

        missing_classes = [c for c in current_targets if c not in classes]
        found_classes = [c for c in current_targets if c in classes]
        for t in found_classes:
            target_images[t] = organelle_arrays[t]

        if len(missing_classes) > 0:
            if not self.use_negative_class:
                raise ValueError(
                    f"Missing classes found, {str(missing_classes)}, please specify use_negative_class=True to generate the missing classes."
                )
            else:
                if len(organelle_arrays) == 0:
                    raise ValueError(
                        f"No target classes found, please specify targets to generate the negative classes."
                    )
                # generate negative class
                if len(organelle_arrays) > 1:
                    found_gt_config = ConcatArrayConfig(
                        name=f"{dataset}_{current_class_name}_{self.output_resolution[0]}nm_gt",
                        channels=list(organelle_arrays.keys()),
                        source_array_configs=organelle_arrays,
                    )
                    missing_mask_config = LogicalOrArrayConfig(
                        name=f"{dataset}_{current_class_name}_{self.output_resolution[0]}nm_labelled_voxels",
                        source_array_config=found_gt_config,
                    )
                else:
                    missing_mask_config = list(organelle_arrays.values())[0]
                missing_gt_config = ConstantArrayConfig(
                    name=f"{dataset}_{current_class_name}_{self.output_resolution[0]}nm_gt",
                    source_array_config=list(organelle_arrays.values())[0],
                    constant=0,
                )
                for t in missing_classes:
                    target_images[t] = missing_gt_config
                    target_masks[t] = missing_mask_config

        for t in found_classes:
            target_masks[t] = ConstantArrayConfig(
                name=f"{dataset}_{t}_{self.output_resolution[0]}nm_labelled_voxels",
                source_array_config=target_images[t],
                constant=1,
            )

        # if len(target_images) > 1:
        gt_config = ConcatArrayConfig(
            name=f"{dataset}_{targets_str}_{self.output_resolution[0]}nm_gt",
            channels=[organelle for organelle in current_targets],
            # source_array_configs={k: gt for k, gt in target_images.items()},
            source_array_configs={k: target_images[k] for k in current_targets},
        )
        mask_config = ConcatArrayConfig(
            name=f"{dataset}_{targets_str}_{self.output_resolution[0]}nm_mask",
            channels=[organelle for organelle in current_targets],
            # source_array_configs={k: mask for k, mask in target_masks.items()},
            # to be sure to have the same order
            source_array_configs={k: target_masks[k] for k in current_targets},
        )
        # else:
        #     gt_config = list(target_images.values())[0]
        #     mask_config = list(target_masks.values())[0]

        return raw_config, gt_config, mask_config

    # @staticmethod
    # def generate_csv(datasets: List[DatasetSpec], csv_path: Path):
    #     print(f"Writing dataspecs to {csv_path}")
    #     with open(csv_path, "w") as f:
    #         for dataset in datasets:
    #             f.write(
    #                 f"{dataset.dataset_type.name},{str(dataset.raw_container)},{dataset.raw_dataset},{str(dataset.gt_container)},{dataset.gt_dataset}\n"
    #             )

    @staticmethod
    def generate_from_csv(
        csv_path: Path,
        input_resolution: Union[Sequence[int], Coordinate],
        output_resolution: Union[Sequence[int], Coordinate],
        name: Optional[str] = None,
        **kwargs,
    ):
        """
        Generate the data split from the CSV file.

        Args:
            csv_path : obj
                The CSV file path.
            input_resolution : obj
                The input resolution.
            output_resolution : obj
                The output resolution.
            name : str
                The name.
            **kwargs : dict
                The keyword arguments.
        Returns:
            obj : The data split.
        Raises:
            FileNotFoundError
            If the file does not exist, a FileNotFoundError is raised.
        Examples:
            >>> generate_from_csv(csv_path, input_resolution, output_resolution, name, **kwargs)
        Notes:
            This function is used to generate the data split from the CSV file.

        """
        if isinstance(csv_path, str):
            csv_path = Path(csv_path)

        if name is None:
            name = csv_path.stem

        return DataSplitGenerator(
            name,
            generate_dataspec_from_csv(csv_path),
            input_resolution,
            output_resolution,
            **kwargs,
        )


def format_class_name(class_name, separator_character="&", targets=None):
    """
    Format the class name.

    Args:
        class_name : obj
            The class name.
        separator_character : str
            The separator character.
    Returns:
        obj : The class name.
    Raises:
        ValueError
            If the class name is invalid, a ValueError is raised.
    Examples:
        >>> format_class_name(class_name, separator_character)
    Notes:
        This function is used to format the class name.
    """
    if "[" in class_name:
        if "]" not in class_name:
            raise ValueError(f"Invalid class name {class_name} missing ']'")
        classes = class_name.split("[")[1].split("]")[0].split(separator_character)
        base_class_name = class_name.split("[")[0]
        return [f"{base_class_name}{c}" for c in classes], classes
    else:
        if targets is None:
            raise ValueError(f"Invalid class name {class_name} missing '[' and ']'")
        if len(targets) > 1:
            raise ValueError(f"Invalid class name {class_name} missing '[' and ']'")
        return [class_name], [targets[0]]
