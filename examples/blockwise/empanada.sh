# This file contains the commands used to run an example of the pretrained empanada segmentation function for mitochondria. This example is based on the jrc_mus-liver-3 dataset.
# The arguments to the segment-blockwise command are:
# -sf: path to the python file with the segmentation_function (in this case the empanada_function.py file)
# -tr: Total ROI to process. It is a list of 3 elements, each one is a list with the start and end of the ROI in the x, y and z axis respectively.
# -rr: ROI to read. It is a list of 3 elements, each one is a list with the start and end of the ROI in the x, y and z axis respectively.
# -wr: ROI to write. It is a list of 3 elements, each one is a list with the start and end of the ROI in the x, y and z axis respectively.
# -nw: Number of workers to use.
# -ic: Input container. It is the path to the input zarr file.
# -id: Input dataset. It is the path to the input dataset inside the input zarr file.
# -oc: Output container. It is the path to the output zarr file.
# -od: Output dataset. It is the path to the output dataset inside the output zarr file.
# --confidence_thr: Confidence threshold to consider a prediction as a mitochondria.
# --center_confidence_thr: Confidence threshold to consider a prediction as a mitochondria center.
# --min_distance_object_centers: Minimum distance between mitochondria centers.
# --median_slices: Number of slices to use for the median filter.
# --min_size: Minimum size of the mitochondria.
# --min_extent: Minimum extent of the mitochondria in a single axis.
# --pixel_vote_thr: Minimum number of votes for a pixel to be considered as a mitochondria pixel (out of 3 for x,y,z).


# jrc_mus-liver-3
dacapo segment-blockwise \
-sf dacapo/blockwise/empanada_function.py \
-tr "[6304:6962, 7234:7632, 9063:9760]" \
-rr "[258,258,258]" \
-wr "[172,172,172]" \
-nw 32 \
-ic data/jrc_mus-liver-3/jrc_mus-liver-3.zarr \
-id recon-1/em/fibsem-uint8/s0 \
-oc predictions/jrc_mus-liver-3/jrc_mus-liver-3.zarr \
-od mito-net-pred \
--confidence_thr 0.73 \
--center_confidence_thr 0.2 \
--min_distance_object_centers 21 \
--median_slices 11 \
--min_size 10000 \
--min_extent 50 \
--pixel_vote_thr 1
