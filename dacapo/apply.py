import logging
from typing import Optional
from funlib.geometry import Roi, Coordinate
import numpy as np
from dacapo.experiments.datasplits.datasets.arrays.array import Array
from dacapo.experiments.datasplits.datasets.dataset import Dataset
from dacapo.experiments.run import Run

from dacapo.experiments.tasks.post_processors.post_processor_parameters import (
    PostProcessorParameters,
)
import dacapo.experiments.tasks.post_processors as post_processors
from dacapo.store.array_store import LocalArrayIdentifier
from dacapo.predict import predict
from dacapo.compute_context import LocalTorch, ComputeContext
from dacapo.experiments.datasplits.datasets.arrays import ZarrArray
from dacapo.store import (
    create_config_store,
    create_weights_store,
)

from pathlib import Path

logger = logging.getLogger(__name__)


def apply(
    run_name: str,
    input_container: Path or str,
    input_dataset: str,
    output_path: Path or str,
    validation_dataset: Optional[Dataset or str] = None,
    criterion: Optional[str] = "voi",
    iteration: Optional[int] = None,
    parameters: Optional[PostProcessorParameters or str] = None,
    roi: Optional[Roi or str] = None,
    num_cpu_workers: int = 30,
    output_dtype: Optional[np.dtype or str] = np.uint8,  # type: ignore
    compute_context: ComputeContext = LocalTorch(),
    overwrite: bool = True,
    file_format: str = "zarr",
):
    """Load weights and apply a model to a dataset. If iteration is None, the best iteration based on the criterion is used. If roi is None, the whole input dataset is used."""
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
    logger.info("Loading run %s", run_name)
    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(run_name)
    run = Run(run_config)

    # create weights store
    weights_store = create_weights_store()

    # load weights
    if iteration is None:
        iteration = weights_store.retrieve_best(run_name, validation_dataset, criterion)  # type: ignore
    logger.info("Loading weights for iteration %i", iteration)
    weights_store.retrieve_weights(run_name, iteration)

    # find the best parameters
    if isinstance(validation_dataset, str) and run.datasplit.validate is not None:
        val_ds_name = validation_dataset
        validation_dataset = [
            dataset for dataset in run.datasplit.validate if dataset.name == val_ds_name
        ][0]
    elif isinstance(validation_dataset, Dataset) or parameters is not None:
        pass
    else:
        raise ValueError(
            "validation_dataset must be a dataset name or a Dataset object, or parameters must be provided explicitly."
        )
    if parameters is None:
        logger.info(
            "Finding best parameters for validation dataset %s", validation_dataset
        )
        parameters = run.task.evaluator.get_overall_best_parameters(  # TODO
            validation_dataset, criterion
        )
        assert (
            parameters is not None
        ), "Unable to retieve parameters. Parameters must be provided explicitly."

    elif isinstance(parameters, str):
        try:
            post_processor_name = parameters.split("(")[0]
            post_processor_kwargs = parameters.split("(")[1].strip(")").split(",")
            post_processor_kwargs = {
                key.strip(): value.strip()
                for key, value in [arg.split("=") for arg in post_processor_kwargs]
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
    input_array_identifier = LocalArrayIdentifier(input_container, input_dataset)
    input_array = ZarrArray.open_from_array_identifier(input_array_identifier)
    if roi is None:
        roi = input_array.roi
    else:
        roi = roi.snap_to_grid(input_array.voxel_size, mode="grow").intersect(
            input_array.roi
        )
    output_container = Path(
        output_path,
        "".join(Path(input_container).name.split(".")[:-1]) + f".{file_format}",
    )
    prediction_array_identifier = LocalArrayIdentifier(
        output_container, f"prediction_{run_name}_{iteration}"
    )
    output_array_identifier = LocalArrayIdentifier(
        output_container, f"output_{run_name}_{iteration}_{parameters}"
    )

    logger.info(
        "Applying best results from run %s at iteration %i to dataset %s",
        run.name,
        iteration,
        Path(input_container, input_dataset),
    )
    return apply_run(
        run,
        parameters,
        input_array,
        prediction_array_identifier,
        output_array_identifier,
        roi,
        num_cpu_workers,
        output_dtype,
        compute_context,
        overwrite,
    )


def apply_run(
    run: Run,
    parameters: PostProcessorParameters,
    input_array: Array,
    prediction_array_identifier: LocalArrayIdentifier,
    output_array_identifier: LocalArrayIdentifier,
    roi: Optional[Roi] = None,
    num_cpu_workers: int = 30,
    output_dtype: Optional[np.dtype] = np.uint8,  # type: ignore
    compute_context: ComputeContext = LocalTorch(),
    overwrite: bool = True,
):
    """Apply the model to a dataset. If roi is None, the whole input dataset is used. Assumes model is already loaded."""
    run.model.eval()

    # render prediction dataset
    logger.info("Predicting on dataset %s", prediction_array_identifier)
    predict(
        run.model,
        input_array,
        prediction_array_identifier,
        output_roi=roi,
        num_cpu_workers=num_cpu_workers,
        output_dtype=output_dtype,
        compute_context=compute_context,
        overwrite=overwrite,
    )

    # post-process the output
    logger.info("Post-processing output to dataset %s", output_array_identifier)
    post_processor = run.task.post_processor
    post_processor.set_prediction(prediction_array_identifier)
    post_processor.process(parameters, output_array_identifier)

    logger.info("Done")
    return
