import funlib.evaluate

import numpy as np
import daisy


def evaluate_detection(
    pred_labels,
    gt_labels,
    return_results=False,
    background_label=0,
    matching_score="overlap",
    matching_threshold=1,
):
    gt_label_data = gt_labels.to_ndarray(roi=pred_labels.roi)
    pred_label_data = pred_labels.to_ndarray(roi=pred_labels.roi)

    # PIXEL-WISE SCORES

    sample_scores = {}

    # accuracy

    sample_scores["accuracy"] = (
        (pred_label_data != background_label) == (gt_label_data != background_label)
    ).sum() / gt_label_data.size
    fg_mask = gt_label_data != background_label
    sample_scores["fg_accuracy"] = (
        (pred_label_data[fg_mask] != background_label)
        == (gt_label_data[fg_mask] != background_label)
    ).sum() / fg_mask.sum()

    # precision, recall, fscore

    relevant = gt_label_data != background_label
    selected = pred_label_data != background_label
    num_relevant = relevant.sum()
    num_selected = selected.sum()

    tp = (pred_label_data[relevant] != background_label).sum()
    fp = num_selected - tp
    tn = (gt_label_data.size - num_relevant) - fp

    # precision, or positive predictive value
    if num_selected > 0:
        ppv = tp / num_selected  # = tp/(tp + fp)
    else:
        ppv = np.nan
    # recall, or true positive rate
    if num_relevant > 0:
        tpr = tp / num_relevant  # = tp/(tp + fn)
    else:
        tpr = np.nan
    # specificity, or true negative rate
    if tn + fp > 0:
        tnr = tn / (tn + fp)
    else:
        tnr = np.nan
    # fall-out, or false positive rate
    if tn + fp > 0:
        fpr = fp / (tn + fp)
    else:
        fpr = np.nan

    if ppv + tpr > 0:
        fscore = 2 * (ppv * tpr) / (ppv + tpr)
    else:
        fscore = np.nan
    balanced_accuracy = (tpr + tnr) / 2

    sample_scores["ppv"] = ppv
    sample_scores["tpr"] = tpr
    sample_scores["tnr"] = tnr
    sample_scores["fpr"] = fpr
    sample_scores["fscore"] = fscore
    sample_scores["balanced_accuracy"] = balanced_accuracy

    # DETECTION SCORES (on foreground objects only)

    detection_scores = funlib.evaluate.detection_scores(
        gt_label_data,
        pred_label_data,
        matching_score=matching_score,
        matching_threshold=matching_threshold,
        voxel_size=pred_labels.voxel_size,
        return_matches=True,
    )
    components = {}
    tp = detection_scores["tp"]
    fp = detection_scores["fp"]
    fn = detection_scores["fn"]
    num_selected = tp + fp
    num_relevant = tp + fn

    # precision, or positive predictive value
    if num_selected > 0:
        ppv = tp / num_selected  # = tp/(tp + fp)
    else:
        ppv = np.nan
    # recall, or true positive rate
    if num_relevant > 0:
        tpr = tp / num_relevant  # = tp/(tp + fn)
    else:
        tpr = np.nan

    if ppv + tpr > 0:
        fscore = 2 * (ppv * tpr) / (ppv + tpr)
    else:
        fscore = np.nan

    sample_scores["detection_ppv"] = ppv
    sample_scores["detection_tpr"] = tpr
    sample_scores["detection_fscore"] = fscore
    sample_scores["avg_iou"] = detection_scores["avg_iou"]

    if return_results:

        components_gt = detection_scores["components_truth"]
        components_pred = detection_scores["components_test"]
        matches = detection_scores["matches"]
        matches_gt = np.array([m[1] for m in matches])
        matches_pred = np.array([m[0] for m in matches])
        components_tp_gt = np.copy(components_gt)
        components_tp_pred = np.copy(components_pred)
        components_fn_gt = np.copy(components_gt)
        components_fp_pred = np.copy(components_pred)
        tp_gt_mask = np.isin(components_gt, matches_gt)
        tp_pred_mask = np.isin(components_pred, matches_pred)
        components_tp_gt[np.logical_not(tp_gt_mask)] = 0
        components_tp_pred[np.logical_not(tp_pred_mask)] = 0
        components_fn_gt[tp_gt_mask] = 0
        components_fp_pred[tp_pred_mask] = 0

        components["volumes/components_tp_gt"] = daisy.Array(
            components_tp_gt, pred_labels.roi, pred_labels.voxel_size
        )
        components["volumes/components_fn_gt"] = daisy.Array(
            components_fn_gt, pred_labels.roi, pred_labels.voxel_size
        )
        components["volumes/components_tp_pred"] = daisy.Array(
            components_tp_pred, pred_labels.roi, pred_labels.voxel_size
        )
        components["volumes/components_fp_pred"] = daisy.Array(
            components_fp_pred, pred_labels.roi, pred_labels.voxel_size
        )

    scores = {"sample": sample_scores, "average": sample_scores}

    if return_results:
        components.update(
            {
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
        )
        return scores, components

    return scores
