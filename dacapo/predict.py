from dacapo.tasks import Task
from dacapo.data import PredictData, Data
from dacapo.predict_pipeline import predict, predict_pipeline

import daisy
import numpy as np

import funlib.run
import torch

from pathlib import Path
import time
import logging


logger = logging.getLogger(__name__)


class PredictRun:
    def __init__(self, run, predict_data, daisy_worker=False, model_padding=None):

        # configs
        self.run = run
        self.predict_data = predict_data
        self.daisy_worker = daisy_worker
        self.model_padding = model_padding

        self.steps = [0]

    def start(self):

        # set torch flags:
        # TODO: make these configurable?
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        self.started = time.time()

        data = PredictData(self.predict_data)
        run_data = Data(self.run.data_config)
        model = self.run.model_config.type(data, self.run.model_config)
        task = Task(run_data, model, self.run.task_config)

        self.__load_best_state(self.run, model, task)

        predict(
            data.raw.test,
            model,
            task.predictor,
            output_dir="test",
            output_filename="data.zarr",
            gt=None,
            aux_tasks=[],
            total_roi=data.total_roi,
            daisy_worker=self.daisy_worker,
            model_padding=self.model_padding,
        )

        self.stopped = time.time()

        logger.info(f"prediction took {self.started - self.stopped} seconds!")

    def __load_best_state(self, run, model, task):
        filename = Path(
            run.outdir,
            f"validation_best_{run.best_score_name}.checkpoint",
        )
        run._load_parameters(filename, model, task.heads, optimizer=None)


def run_local(run, data, daisy_config):
    predict = PredictRun(
        run, data, daisy_config.worker, getattr(daisy_config, "model_padding", None)
    )

    print(f"Running run {predict.run.hash} with data {predict.predict_data}")

    predict.start()


def process_block(outdir, block, fail=None):

    logger.debug("Processing block %s", block)

    if block.block_id == fail:
        raise RuntimeError("intended failure")

    path = Path(outdir, "%d.block" % block.block_id)
    with open(path, "w") as f:
        f.write(str(block.block_id))

    return 0


def predict_worker(dacapo_flags, bsub_flags, outdir):

    worker_id = daisy.Context.from_env().worker_id

    log_out = f"{outdir}/out_{worker_id}.log"
    log_err = f"{outdir}/out_{worker_id}.err"
    command = [f"dacapo predict {dacapo_flags}"]

    daisy.call(command, log_out=log_out, log_err=log_err)

    logging.warning("Predict worker finished")


def run_remote(run, data, daisy_config, dacapo_flags, bsub_flags):
    training_data = Data(run.data_config)
    predict_data = PredictData(data)
    model = run.model_config.type(training_data, run.model_config)
    task = Task(training_data, model, run.task_config)
    post_processor = task.predictor.post_processor

    # build pipeline to prepare datasets
    compute_pipeline, sources, total_request = predict_pipeline(
        predict_data.raw,
        model,
        task.predictor,
        output_dir="test",
        output_filename="data.zarr",
        gt=None,
        aux_tasks=task.aux_tasks,
        total_roi=predict_data.total_roi,
        model_padding=getattr(daisy_config, "model_padding", None),
        daisy_worker=False,
    )

    outdir = "test"
    if not Path(outdir).exists():
        Path(outdir).mkdir()

    voxel_size = np.array(predict_data.raw.test.voxel_size, dtype=int)
    input_shape = np.array(model.input_shape, dtype=int) * voxel_size
    output_shape = np.array(model.output_shape, dtype=int) * voxel_size
    context = (input_shape - output_shape) // 2
    offset = input_shape * 0
    if hasattr(daisy_config, "model_padding"):
        model_padding = daisy_config.model_padding * voxel_size

        input_block_roi = daisy.Roi(tuple(offset), tuple(input_shape + model_padding))
        output_block_roi = daisy.Roi(
            tuple(context), tuple(output_shape + model_padding)
        )
    else:
        input_block_roi = daisy.Roi(tuple(offset), tuple(input_shape))
        output_block_roi = daisy.Roi(tuple(context), tuple(output_shape))

    logger.warning("Starting blockwise prediction")
    daisy.run_blockwise(
        daisy.Roi(predict_data.raw.roi.get_offset(), predict_data.raw.roi.get_shape()),
        input_block_roi,
        output_block_roi,
        process_function=lambda: predict_worker(dacapo_flags, bsub_flags, outdir),
        num_workers=daisy_config.num_workers,
        read_write_conflict=False,
        fit="overhang",
    )
    logger.warning("Finished blockwise prediction")

    for name, fit, step in post_processor.daisy_steps():
        post_processing_parameters = post_processor.daisy_parameters[name]
        logger.warning(f"Starting blockwise {name}")
        daisy.run_blockwise(
            daisy.Roi(
                predict_data.raw.roi.get_offset(), predict_data.raw.roi.get_shape()
            ),
            input_block_roi,
            output_block_roi,
            process_function=lambda: step(**post_processing_parameters),
            num_workers=daisy_config.num_workers,
            read_write_conflict=False,
            fit=fit,
        )
        logger.warning(f"Finished blockwise {name}")
