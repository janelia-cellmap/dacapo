# %%
from shared import (
    config_store,
    trainer_config,
    input_resolution,
    output_resolution,
    architecture_config,
    repetitions,
    iterations,
    validation_interval,
    run_command,
    DataSplitGenerator,
    RunConfig,
    Run,
    subprocess,
)
from dacapo.experiments.tasks import OneHotTaskConfig

datasplit_config = DataSplitGenerator.generate_from_csv(
    "/nrs/cellmap/rhoadesj/dacapo/experiments/hot_distance/hot_distance_datasplit.csv",
    input_resolution,
    output_resolution,
).compute()

datasplit = datasplit_config.datasplit_type(datasplit_config)
viewer = datasplit._neuroglancer()
config_store.store_datasplit_config(datasplit_config)

task_type = "onehot"

task_config = OneHotTaskConfig(name="onehot_task_4nm", classes=["mito"])
config_store.store_task_config(task_config)


# %%

for i in range(repetitions):
    run_name = ("_").join(
        [
            "onehot",
            datasplit_config.name,
            task_config.name,
            architecture_config.name,
            trainer_config.name,
        ]
    ) + f"__{i}"
    run_config = RunConfig(
        name=run_name,
        datasplit_config=datasplit_config,
        task_config=task_config,
        architecture_config=architecture_config,
        trainer_config=trainer_config,
        num_iterations=iterations,
        validation_interval=validation_interval,
        repetition=i,
        start_config=None,
    )

    print(run_config.name)
    config_store.store_run_config(run_config)
    run = Run(config_store.retrieve_run_config(run_name))

    # run in parallel
    subprocess.run(run_command.format(run_name=run_name, task_type=task_type))
