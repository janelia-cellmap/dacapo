import dacapo
from dacapo.store.create_store import create_config_store

config_store = create_config_store()
run_config = config_store.retrieve_run_config("cosem_distance_run_4nm_finetune3")

from dacapo.experiments.run import Run

run = Run(run_config)
from dacapo.train import train_run

train_run(run)
