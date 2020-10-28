from dacapo.tasks import Task
from dacapo.data import PredictData
from dacapo.predict_pipeline import predict

import funlib.run
import zarr
import torch

from pathlib import Path
import time
import logging


logger = logging.getLogger(__name__)


class PredictRun:
    def __init__(
        self,
        run,
        predict_data,
    ):

        # configs
        self.run = run
        self.predict_data = predict_data

    def start(self):

        # set torch flags:
        # TODO: make these configurable?
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        self.started = time.time()

        data = PredictData(self.predict_data)
        model = self.model_config.type(data, self.run.model_config)
        task = Task(data, model, self.run.task_config)

        self.__load_best_state(self.run, model, task)

        results = predict(
            data.raw.test, model, task.predictor, gt=None, aux_tasks=task.aux_tasks
        )

        output_container = zarr.open(data.raw.test.filename)
        prefix = data.raw.test.ds_name
        for k, v in results:
            output_container[f"{prefix}_{k}"] = v.data
            output_container[f"{prefix}_{k}"].attrs["offset"] = v.spec.roi.get_offset()
            output_container[f"{prefix}_{k}"].attrs["resolution"] = v.spec.voxel_size

        self.stopped = time.time()

        logger.info(f"prediction took {self.started - self.stopped} seconds!")

    def __load_best_state(self, run, model, task):
        filename = Path(
            run.outdir,
            f"validation_best_{run.best_score_name}.checkpoint",
        )
        run._load_parameters(filename, model, task.heads, optimizer=None)


def run_local(run, data):
    predict = PredictRun(run, data)

    print(f"Running run {predict.run.hash} with data {predict.predict_data}")

    predict.start()


def run_remote(run):
    if run.billing is not None:
        flags = [f"-P {run.billing}"]
    else:
        flags = None

    funlib.run.run(
        command=f"dacapo run-one "
        f"-t {run.task_config.config_file} "
        f"-d {run.data_config.config_file} "
        f"-m {run.model_config.config_file} "
        f"-o {run.optimizer_config.config_file} "
        f"-R {run.repetition} "
        f"-v {run.validation_interval} "
        f"-s {run.snapshot_interval} "
        f"-b {run.keep_best_validation} ",
        num_cpus=2,
        num_gpus=1,
        queue="gpu_any",
        execute=True,
        flags=flags,
        batch=run.batch,
        log_file=f"runs/{run.hash}/log.out",
        error_file=f"runs/{run.hash}/log.err",
    )
