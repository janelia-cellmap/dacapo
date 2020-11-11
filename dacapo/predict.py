from dacapo.tasks import Task
from dacapo.data import PredictData, Data
from dacapo.predict_pipeline import predict

import daisy

import funlib.run
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
        daisy=False,
    ):

        # configs
        self.run = run
        self.predict_data = predict_data

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
            gt=None,
            aux_tasks=[],
            total_roi=data.total_roi,
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
    predict = PredictRun(run, data, daisy_config)

    print(f"Running run {predict.run.hash} with data {predict.predict_data}")

    predict.start()


def predict_worker(dacapo_flags, bsub_flags):

    funlib.run.run(
        command=f"dacapo predict {dacapo_flags}",
        num_cpus=2,
        num_gpus=1,
        queue="gpu_any",
        execute=True,
        flags=bsub_flags,
        batch=True,
        log_file="predict_logs/%J.log",
        error_file="predict_logs/%J.err",
    )


def run_remote(dacapo_flags, bsub_flags, daisy_config):

    daisy.run_blockwise(
        daisy.Roi(*daisy_config.total_roi),
        daisy.Roi(*daisy_config.input_block_roi),
        daisy.Roi(*daisy_config.output_block_roi),
        process_function=lambda: predict_worker(dacapo_flags, bsub_flags),
        check_function=lambda b: False,
        num_workers=daisy_config.num_workers,
        read_write_conflict=False,
        fit="overhang",
    )
