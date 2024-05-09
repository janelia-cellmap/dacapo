# %%
import dacapo

# from  import create_config_store

config_store = dacapo.store.create_store.create_config_store()

# %%
from dacapo import Options

options = Options.instance()

# %%
options
# %%
from dacapo.experiments.tasks import DistanceTaskConfig

task_config = DistanceTaskConfig(
    name="cosem_distance_task_4nm",
    channels=["mito"],
    clip_distance=40.0,
    tol_distance=40.0,
    scale_factor=80.0,
)

# %%

config_store.store_task_config(task_config)

# %%
