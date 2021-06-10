from funlib.evaluate import rand_voi

import numpy as np
import daisy


def evaluate_affs(pred_labels, gt_labels, return_results=False):
    gt_label_data = gt_labels.to_ndarray(roi=pred_labels.roi)
    pred_label_data = pred_labels.to_ndarray(roi=pred_labels.roi)

    results = rand_voi(gt_label_data, pred_label_data)
    results["voi_sum"] = results["voi_split"] + results["voi_merge"]

    scores = {"sample": results, "average": results}

    if return_results:
        results = {
            "volumes/pred_labels": daisy.Array(
                pred_labels.data.astype(np.uint64),
                roi=pred_labels.roi,
                voxel_size=pred_labels.voxel_size,
            ),
            "volumes/gt_labels": daisy.Array(
                gt_labels.data.astype(np.uint64),
                roi=gt_labels.roi,
                voxel_size=gt_labels.voxel_size,
            ),
        }

        return scores, results

    return scores
