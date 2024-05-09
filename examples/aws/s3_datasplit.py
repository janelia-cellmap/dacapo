# %%
from dacapo.experiments.datasplits import DataSplitGenerator
from funlib.geometry import Coordinate

input_resolution = Coordinate(8, 8, 8)
output_resolution = Coordinate(4, 4, 4)
datasplit_config = DataSplitGenerator.generate_from_csv(
    "cloud_csv.csv",
    input_resolution,
    output_resolution,
).compute()
# %%
datasplit = datasplit_config.datasplit_type(datasplit_config)
# %%
viewer = datasplit._neuroglancer()
# %%
