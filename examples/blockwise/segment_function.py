import logging

import scipy.ndimage
import yaml

logger = logging.getLogger(__file__)


def segment_function(input_array, block, config_path):
    """
    Segment a 3D block using a small numpy-based post-processing script.

    Args:
        input_array (np.ndarray): The 3D array to segment.
        block (daisy.Block): The block object.
        config_path (str): The path to the configuration yaml file.
    Returns:
        np.ndarray: The segmented 3D array.
    """
    data = input_array.to_ndarray(block.read_roi)
    steps = yaml.load(config_path, Loader=yaml.FullLoader)

    # Apply the segmentation function here
    for step, params in steps.items():
        if step == "gaussian_smooth":
            sigma = params.get("sigma", 1.0)
            logger.info(f"Applying Gaussian smoothing with sigma={sigma}")
            data = scipy.ndimage.gaussian_filter(data, sigma=sigma)
        elif step == "threshold":
            threshold = params.get("threshold", 0.5)
            logger.info(f"Applying thresholding with threshold={threshold}")
            data = data > threshold
        elif step == "label":
            structuring_element = params.get("structuring_element", None)
            logger.info("Applying labeling")
            data, _ = scipy.ndimage.label(data, structuring_element)  # type: ignore

    return data
