import logging
from typing import Optional
from funlib.geometry import Roi, Coordinate
from funlib.persistence import open_ds
import numpy as np
from dacapo.experiments.datasplits.datasets.dataset import Dataset
from dacapo.experiments.run import Run

from dacapo.experiments.tasks.post_processors.post_processor_parameters import (
    PostProcessorParameters,
)
import dacapo.experiments.tasks.post_processors as post_processors
from dacapo.store.array_store import LocalArrayIdentifier
from dacapo.predict import predict
from dacapo.store.create_store import (
    create_config_store,
    create_weights_store,
)

from upath import UPath as Path

logger = logging.getLogger(__name__)


def apply(
    run_name: str,
    input_container: Path | str,
    input_dataset: str,
    output_path: Path | str,
    validation_dataset: Optional[Dataset | str] = None,
    criterion: str = "voi",
    iteration: Optional[int] = None,
    parameters: Optional[PostProcessorParameters | str] = None,
    roi: Optional[Roi | str] = None,
    num_workers: int = 12,
    output_dtype: np.dtype | str = np.uint8,  # type: ignore
    overwrite: bool = True,
    file_format: str = "zarr",
):
    """
    Load weights and apply a trained model to a dataset. If iteration is None, the best iteration based on the criterion is used. If roi is None, the whole input dataset is used.

    Args:
        run_name (str): Name of the run to apply.
        input_container (Path | str): Path to the input container.
        input_dataset (str): Name of the input dataset.
        output_path (Path | str): Path to the output container.
        validation_dataset (Optional[Dataset | str], optional): Validation dataset to use for finding the best parameters. Defaults to None.
        criterion (str, optional): Criterion to use for finding the best parameters. Defaults to "voi".
        iteration (Optional[int], optional): Iteration to use. If None, the best iteration is used. Defaults to None.
        parameters (Optional[PostProcessorParameters | str], optional): Post-processor parameters to use. If None, the best parameters are found. Defaults to None.
        roi (Optional[Roi | str], optional): Region of interest to use. If None, the whole input dataset is used. Defaults to None.
        num_workers (int, optional): Number of workers to use. Defaults to 12.
        output_dtype (np.dtype | str, optional): Output dtype. Defaults to np.uint8.
        overwrite (bool, optional): Overwrite existing output. Defaults to True.
        file_format (str, optional): File format to use. Defaults to "zarr".
    Raises:
        ValueError: If validation_dataset is None and criterion is not None.
        ValueError: If parameters is a string that cannot be parsed to PostProcessorParameters.
        ValueError: If parameters is not a PostProcessorParameters object.
    Examples:
        >>> apply(
        ...     run_name="run_1",
        ...     input_container="data.zarr",
        ...     input_dataset="raw",
        ...     output_path="output.zarr",
        ...     validation_dataset="validate",
        ...     criterion="voi",
        ...     num_workers=12,
        ...     output_dtype=np.uint8,
        ...     overwrite=True,
        ... )
    """
    if isinstance(output_dtype, str):
        output_dtype = np.dtype(output_dtype)

    if isinstance(roi, str):
        start, end = zip(
            *[
                tuple(int(coord) for coord in axis.split(":"))
                for axis in roi.strip("[]").split(",")
            ]
        )
        roi = Roi(
            Coordinate(start),
            Coordinate(end) - Coordinate(start),
        )

    assert (validation_dataset is not None and isinstance(criterion, str)) or (
        isinstance(iteration, int)
    ), "Either validation_dataset and criterion, or iteration must be provided."

    # retrieving run
    print(f"Loading run {run_name}")
    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(run_name)
    run = Run(run_config)

    # create weights store
    weights_store = create_weights_store()

    # load weights
    if iteration is None:
        iteration = weights_store.retrieve_best(run_name, validation_dataset, criterion)  # type: ignore
    print(f"Loading weights for iteration {iteration}")
    weights_store.retrieve_weights(run_name, iteration)

    if parameters is None:
        # find the best parameters
        _validation_dataset: Dataset
        if isinstance(validation_dataset, str) and run.datasplit.validate is not None:
            val_ds_name = validation_dataset
            _validation_dataset = [
                dataset
                for dataset in run.datasplit.validate
                if dataset.name == val_ds_name
            ][0]
        elif isinstance(validation_dataset, Dataset):
            _validation_dataset = validation_dataset
        else:
            raise ValueError(
                "validation_dataset must be a dataset name or a Dataset object, or parameters must be provided explicitly."
            )
        print(f"Finding best parameters for validation dataset {_validation_dataset}")
        parameters = run.task.evaluator.get_overall_best_parameters(
            _validation_dataset, criterion
        )
        assert (
            parameters is not None
        ), "Unable to retieve parameters. Parameters must be provided explicitly."

    elif isinstance(parameters, str):
        try:
            post_processor_name = parameters.split("(")[0]
            _post_processor_kwargs = parameters.split("(")[1].strip(")").split(",")
            post_processor_kwargs = {
                key.strip(): value.strip()
                for key, value in [arg.split("=") for arg in _post_processor_kwargs]
            }
            for key, value in post_processor_kwargs.items():
                if value.isdigit():
                    post_processor_kwargs[key] = int(value)  # type: ignore
                elif value.replace(".", "", 1).isdigit():
                    post_processor_kwargs[key] = float(value)  # type: ignore
        except:
            raise ValueError(
                f"Could not parse parameters string {parameters}. Must be of the form 'post_processor_name(arg1=val1, arg2=val2, ...)'"
            )
        try:
            parameters = getattr(post_processors, post_processor_name)(
                **post_processor_kwargs
            )
        except Exception as e:
            logger.error(
                f"Could not instantiate post-processor {post_processor_name} with arguments {post_processor_kwargs}.",
                exc_info=True,
            )
            raise e

    assert isinstance(
        parameters, PostProcessorParameters
    ), "Parameters must be parsable to a PostProcessorParameters object."

    # make array identifiers for input, predictions and outputs
    input_array_identifier = LocalArrayIdentifier(Path(input_container), input_dataset)
    input_array = open_ds(
        f"{input_array_identifier.container}/{input_array_identifier.dataset}"
    )
    if roi is None:
        _roi = input_array.roi
    else:
        _roi = roi.snap_to_grid(input_array.voxel_size, mode="grow").intersect(
            input_array.roi
        )
    output_container = Path(
        output_path,
        Path(input_container).stem + f".{file_format}",
    )
    prediction_array_identifier = LocalArrayIdentifier(
        output_container, f"prediction_{run_name}_{iteration}"
    )
    output_array_identifier = LocalArrayIdentifier(
        output_container, f"output_{run_name}_{iteration}_{parameters}"
    )

    print(
        f"Applying best results from run {run.name} at iteration {iteration} to dataset {Path(input_container, input_dataset)}"
    )
    return apply_run(
        run,
        iteration,
        parameters,
        input_array_identifier,
        prediction_array_identifier,
        output_array_identifier,
        _roi,
        num_workers,
        output_dtype,
        overwrite,
    )


def apply_run(
    run: Run,
    iteration: int,
    parameters: PostProcessorParameters,
    input_array_identifier: "LocalArrayIdentifier",
    prediction_array_identifier: "LocalArrayIdentifier",
    output_array_identifier: "LocalArrayIdentifier",
    roi: Optional[Roi] = None,
    num_workers: int = 12,
    output_dtype: np.dtype | str = np.uint8,  # type: ignore
    overwrite: bool = True,
):
    """
    Apply the model to a dataset. If roi is None, the whole input dataset is used. Assumes model is already loaded.

    Args:
        run (Run): The run object containing the task and post-processor.
        iteration (int): The iteration number.
        parameters (PostProcessorParameters): The post-processor parameters.
        input_array_identifier (LocalArrayIdentifier): The identifier for the input array.
        prediction_array_identifier (LocalArrayIdentifier): The identifier for the prediction array.
        output_array_identifier (LocalArrayIdentifier): The identifier for the output array.
        roi (Optional[Roi], optional): The region of interest. Defaults to None.
        num_workers (int, optional): The number of workers for parallel processing. Defaults to 12.
        output_dtype (np.dtype | str, optional): The output data type. Defaults to np.uint8.
        overwrite (bool, optional): Whether to overwrite existing output. Defaults to True.
    Raises:
        ValueError: If the input array is not a zarr array.
    Examples:
        >>> apply_run(
        ...     run=run,
        ...     iteration=1,
        ...     parameters=parameters,
        ...     input_array_identifier=LocalArrayIdentifier(Path("data.zarr"), "raw"),
        ...     prediction_array_identifier=LocalArrayIdentifier(Path("output.zarr"), "prediction_run_1_1"),
        ...     output_array_identifier=LocalArrayIdentifier(Path("output.zarr"), "output_run_1_1"),
        ...     roi=None,
        ...     num_workers=12,
        ...     output_dtype=np.uint8,
        ...     overwrite=True,
        ... )
    """
    # render prediction dataset
    print(f"Predicting on dataset {prediction_array_identifier}")
    predict(
        run.name,
        iteration,
        input_container=input_array_identifier.container,
        input_dataset=input_array_identifier.dataset,
        output_path=prediction_array_identifier,
        output_roi=roi,
        num_workers=num_workers,
        output_dtype=output_dtype,
        overwrite=overwrite,
    )

    # post-process the output
    print(
        f"Post-processing output to dataset {output_array_identifier}",
        output_array_identifier,
    )
    post_processor = run.task.post_processor
    post_processor.set_prediction(prediction_array_identifier)
    post_processor.process(parameters, output_array_identifier, num_workers=num_workers)

    print("Done")
    return
