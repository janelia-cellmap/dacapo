# This file contains the commands used to run an example of the segment-blockwise command.
# The arguments to the segment-blockwise command are:
# -sf: path to the python file with the segmentation_function (in this case the empanada_function.py file)
# -tr: Total ROI to process. It is a list of 3 elements, each one is a list with the start and end of the ROI in the x, y and z axis respectively.
# e.g. -tr "[320000:330000, 100000:110000, 10000:20000]" \
# -rr: ROI to read. It is a list of 3 elements, each one is a list with the start and end of the ROI in the x, y and z axis respectively.
# -wr: ROI to write. It is a list of 3 elements, each one is a list with the start and end of the ROI in the x, y and z axis respectively.
# -nw: Number of workers to use.
# -ic: Input container. It is the path to the input zarr file.
# -id: Input dataset. It is the path to the input dataset inside the input zarr file.
# -oc: Output container. It is the path to the output zarr file.
# -od: Output dataset. It is the path to the output dataset inside the output zarr file.
# --config_path: Path to the config yaml file.


dacapo segment-blockwise \
-sf segment_function.py \
-rr "[256,256,256]" \
-wr "[256,256,256]" \
-nw 32 \
-ic predictions/c-elegen/bw/c_elegen_bw_op50_ld_scratch_0_300000.zarr \
-id ld/ld \
-oc predictions/c-elegen/bw/jrc_c-elegans-bw-1_postprocessed.zarr \
-od ld \
--config_path segment_config.yaml
