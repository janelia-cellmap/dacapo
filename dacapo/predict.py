from dacapo.tasks import Task
from dacapo.data import DataSource
from dacapo.store import MongoDbStore
from .daisy import predict

import daisy
from funlib.geometry import Coordinate, Roi
import funlib.run
import attr

from dacapo.data import Dataset, ArrayDataSource
from dacapo.models import Model

from pathlib import Path
from typing import List, Optional
import logging
import subprocess
import os


logger = logging.getLogger(__name__)


@attr.s
class PredictConfig:
    run_id: str = attr.ib()
    pred_id: str = attr.ib()
    dataset_id: str = attr.ib()
    data_source: str = attr.ib()
    out_container: Path = attr.ib()
    backbone_checkpoint: Path = attr.ib()
    head_checkpoints: List[Path] = attr.ib()
    gt: Optional[ArrayDataSource] = attr.ib()
    raw: ArrayDataSource = attr.ib()
    input_shape: Coordinate = attr.ib()
    output_shape: Coordinate = attr.ib()
    num_workers: int = attr.ib(default=1)

    @property
    def output_dir(self):
        return Path(self.out_container).parent

    @property
    def output_filename(self):
        return Path(self.out_container).name

    @property
    def voxel_size(self):
        assert self.gt.voxel_size == self.raw.voxel_size
        return self.raw.voxel_size

    @property
    def input_size(self):
        return self.input_shape * self.voxel_size

    @property
    def output_size(self):
        return self.output_shape * self.voxel_size

    @property
    def context(self):
        return (self.input_size - self.output_size) / 2

    @property
    def input_roi(self):
        if self.gt is None:
            input_roi = self.raw.roi
        else:
            output_roi = self.gt.roi
            input_roi = output_roi.grow(self.context, self.context)
        return input_roi

    @property
    def output_roi(self):
        if self.gt is None:
            input_roi = self.raw.roi
            output_roi = input_roi.grow(-self.context, -self.context)
        else:
            output_roi = self.gt.roi
        return output_roi

    @property
    def task_id(self):
        return f"{self.run_id}_{self.pred_id}"

    def task(self):
        store = MongoDbStore()

        print(
            {
                "total_roi": self.input_roi,
                "read_roi": Roi((0,) * self.input_size.dims, self.input_size),
                "write_roi": Roi(self.context, self.output_size),
            }
        )
        return daisy.Task(
            task_id=self.task_id,
            total_roi=self.input_roi,
            read_roi=Roi((0,) * self.input_size.dims, self.input_size),
            write_roi=Roi(self.context, self.output_size),
            process_function=remote_prediction_worker(
                self.run_id,
                self.pred_id,
                self.dataset_id,
                self.data_source,
                self.out_container,
                self.backbone_checkpoint,
                self.head_checkpoints,
            ),
            check_function=lambda b: store.check_block(
                self.task_id, "prediction", b.block_id
            ),
            num_workers=self.num_workers,
            fit="overhang",
        )


def load_prediction_config(
    run_id,
    prediction_id,
    dataset_id,
    data_source,
    out_container,
    backbone_checkpoint,
    head_checkpoints,
):
    # Load Predict Config from cli arguments
    store = MongoDbStore()
    # get necessary configs from job
    run = store.get_run(run_id)
    dataset = store.get_dataset(run.dataset)
    model = store.get_model(run.model)

    input_shape = Coordinate(model.input_shape)
    output_shape = (
        Coordinate(model.output_shape) if model.output_shape is not None else None
    )

    backbone = model.instantiate(dataset)
    if output_shape is None:
        output_shape = Coordinate(backbone.output_shape(input_shape))

    # load in data to run on
    dataset = store.get_dataset(dataset_id)
    assert hasattr(dataset, "raw") and hasattr(
        dataset.raw, data_source
    ), f"{dataset} has no raw.{data_source}"
    raw = getattr(dataset.raw, data_source)
    try:
        gt = getattr(dataset.gt, data_source)
    except AttributeError:
        gt = None

    # switch to world units
    voxel_size = raw.voxel_size
    input_size = voxel_size * input_shape
    output_size = voxel_size * output_shape

    # calculate input and output rois
    context = (input_size - output_size) / 2
    logger.warning(f"context: {context}")

    if gt is None:
        input_roi = raw.roi
        output_roi = input_roi.grow(-context, -context)
    else:
        output_roi = gt.roi
        input_roi = output_roi.grow(context, context)

    return PredictConfig(
        run_id,
        prediction_id,
        dataset_id,
        data_source,
        out_container,
        backbone_checkpoint,
        head_checkpoints,
        gt,
        raw,
        input_shape,
        output_shape,
    )


def predict_one(
    run_id,
    prediction_id,
    dataset_id,
    data_source,
    out_container,
    backbone_checkpoint,
    head_checkpoints,
):
    # Command line entrypoint to generating a prediction.
    # This is a blocking call, will only return when prediction
    # is completed.
    # This call creates a daisy task and spins up a set of predict workers

    # There is no support for predicting on a sub_roi of a volumes yet.
    # If you need this, you should copy the data you want to predict into a
    # seperate zarr dataset and then predict on it.
    logger.warning("Predicting on validation data...")

    Path(out_container).mkdir(parents=True, exist_ok=True)

    predict_config = load_prediction_config(
        run_id,
        prediction_id,
        dataset_id,
        data_source,
        out_container,
        backbone_checkpoint,
        head_checkpoints,
    )

    predict_task = predict_config.task()
    return daisy.run_blockwise([predict_task])


def predict_worker(
    run_id,
    prediction_id,
    dataset_id,
    data_source,
    out_container,
    backbone_checkpoint,
    head_checkpoints,
):
    logger.warning("Predicting in worker... ")

    Path(out_container).mkdir(parents=True, exist_ok=True)

    logger.warning("Loading config... ")

    predict_config = load_prediction_config(
        run_id,
        prediction_id,
        dataset_id,
        data_source,
        out_container,
        backbone_checkpoint,
        head_checkpoints,
    )
    logger.warning("Config loaded... ")
    predict(predict_config)


def remote_prediction_worker(
    run_id,
    prediction_id,
    dataset_id,
    data_source,
    out_container,
    backbone_checkpoint,
    head_checkpoints,
):
    store = MongoDbStore()
    run = store.get_run(run_id)
    out_dir = Path(out_container).parent

    command_args = funlib.run.run(
        command=f"dacapo predict-worker -r {run_id} -p {prediction_id} "
        f"-d {dataset_id} -ds {data_source} -oc {out_container} "
        f"-bb {backbone_checkpoint} "
        + " ".join(f"-hs {head_checkpoint}" for head_checkpoint in head_checkpoints),
        num_cpus=5,
        num_gpus=1,
        queue="gpu_rtx",
        execute=False,
        expand=False,
        flags=run.execution_details.bsub_flags.split(" "),
        batch=run.execution_details.batch,
        log_file=f"{out_dir}/log.out",
        error_file=f"{out_dir}/log.err",
    )

    def process_function():
        print("starting worker:")
        subprocess.run(" ".join(command_args), shell=True, encoding="UTF-8")

    return process_function
