from funlib.evaluate import rand_voi

import numpy as np
import gunpowder as gp


def evaluate_affs(pred_labels, gt_labels, return_results=False):

    results = rand_voi(gt_labels.data, pred_labels.data)
    results["voi_sum"] = results["voi_split"] + results["voi_merge"]
    
    scores = {"sample": results, "average": results}

    if return_results:
        results = {
            "pred_labels": gp.Array(pred_labels.data.astype(np.uint64), gp.ArraySpec(roi=pred_labels.spec.roi, voxel_size = pred_labels.spec.voxel_size)),
            "gt_labels": gp.Array(gt_labels.data.astype(np.uint64), gp.ArraySpec(roi=gt_labels.spec.roi, voxel_size = gt_labels.spec.voxel_size)),
        }

        return scores, results

    return scores
