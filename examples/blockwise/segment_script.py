from dacapo.blockwise.scheduler import segment_blockwise
from funlib.geometry import Roi, Coordinate

# Make the ROIs
path_to_function = "segment_function.py"
context = Coordinate((0, 0, 0))
total_roi = Roi(offset=(0, 0, 0), shape=(100, 100, 100))
read_roi = Roi(offset=(0, 0, 0), shape=(10, 10, 10))
write_roi = Roi(offset=(0, 0, 0), shape=(1, 1, 1))
num_workers = 16

# Run the script blockwise
success = segment_blockwise(
    segment_function_file=path_to_function,
    context=context,
    total_roi=total_roi,
    read_roi=read_roi,
    write_roi=write_roi,
    num_workers=num_workers,
    steps={
        "gaussian_smooth": {"sigma": 1.0},
        "threshold": {"threshold": 0.5},
        "label": {},
    },
)

# Print the success
if success:
    print("Success")
else:
    print("Failure")

# example run command:
# bsub -n 4 python dummy_script.py
