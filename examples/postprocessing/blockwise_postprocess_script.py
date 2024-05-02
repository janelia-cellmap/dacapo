from dacapo.blockwise.scheduler import run_blockwise
from funlib.geometry import Roi
from postprocessing.postprocess_worker import open_ds
import daisy
import numpy as np

# Make the ROIs
path_to_worker = "postprocess_worker.py"
num_workers = 16
overlap = 20

peroxi_container = "/path/to/peroxi_container.zarr"
peroxi_dataset = "peroxisomes"
mito_container = "/path/to/mito_container.zarr"
mito_dataset = "mitochondria"
threshold = "0.5"
gaussian_kernel = 2

array_in = open_ds(peroxi_container, peroxi_dataset)
total_roi = array_in.roi

voxel_size = array_in.voxel_size
block_size = np.array(array_in.data.chunks) * np.array(voxel_size)

write_size = daisy.Coordinate(block_size)
write_roi = daisy.Roi((0,) * len(write_size), write_size)

context = np.array(voxel_size) * overlap

read_roi = write_roi.grow(context, context)
total_roi = array_in.roi.grow(context, context)


# Run the script blockwise
success = run_blockwise(
    worker_file=path_to_worker,
    total_roi=total_roi,
    read_roi=read_roi,
    write_roi=write_roi,
    num_workers=num_workers,
)

# Print the success
if success:
    print("Success")
else:
    print("Failure")

# example run command:
# bsub -n 4 python blockwise_postprocess_script.py
