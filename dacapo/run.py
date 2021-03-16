import funlib.run

from dacapo.configs import Run
from dacapo.store import MongoDbStore

import logging

logger = logging.getLogger(__name__)


def enumerate_runs(
    task_configs,
    data_configs,
    model_configs,
    optimizer_configs,
    repetitions,
    num_iterations,
    validation_interval,
    snapshot_interval,
    keep_best_validation,
    bsub_flags,
    batch,
):

    runs = []
    for task_config in task_configs:
        for data_config in data_configs:
            for model_config in model_configs:
                for optimizer_config in optimizer_configs:
                    for repetition in range(repetitions):
                        runs.append(
                            Run(
                                task_config,
                                data_config,
                                model_config,
                                optimizer_config,
                                repetition,
                                num_iterations,
                                validation_interval,
                                snapshot_interval,
                                keep_best_validation,
                                bsub_flags=bsub_flags,
                                batch=batch,
                            )
                        )
    return runs


def run_local(run):
    store = MongoDbStore()
    store.sync_run(run)
    if not run.outdir.exists():
        run.outdir.mkdir(parents=True, exist_ok=True)

    print(
        f"Running task {run.task} "
        f"with data {run.dataset}, "
        f"with model {run.model}, "
        f"using optimizer {run.optimizer}"
    )

    run.start()


def run_remote(run):
    store = MongoDbStore()
    store.sync_run(run)
    print(f"synced run with id {run.id}")
    if not run.outdir.exists():
        run.outdir.mkdir(parents=True, exist_ok=True)

    funlib.run.run(
        command=f"dacapo run-one -r {run.id}",
        num_cpus=5,
        num_gpus=1,
        queue="gpu_rtx",
        execute=True,
        flags=run.execution_details.bsub_flags.split(" "),
        batch=run.execution_details.batch,
        log_file=f"{run.outdir}/log.out",
        error_file=f"{run.outdir}/log.err",
    )


def run_all(runs, num_workers=None):

    print(f"Running {len(runs)} configs:")
    for run in runs[:10]:
        print(f"\t{run}")
    if len(runs) > 10:
        print(f"(and {len(runs) - 10} more...)")

    if num_workers is not None:

        from multiprocessing import Pool

        with Pool(num_workers) as pool:
            pool.map(run_remote, runs)

    else:

        for run in runs:
            run_local(run)
