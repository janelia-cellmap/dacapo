csv_path = "/groups/cellmap/cellmap/zouinkhim/dacapo_release/tmp/dacapo/dacapo/experiments/datasplits/test.csv"

from datasplit_generator import DataSplitGenerator
from funlib.geometry import Coordinate

input_resolution = Coordinate(16,16,16)
output_resolution = Coordinate(8,8,8)
datasplit_config = DataSplitGenerator.generate_from_csv(csv_path, input_resolution, output_resolution).compute()

# print(datasplit)
datasplit = datasplit_config.datasplit_type(datasplit_config)
viewer  = datasplit._neuroglancer_link()

print(viewer)
input()
