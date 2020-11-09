import funlib.evaluate
import numpy as np

import gunpowder as gp


def evaluate_labels(
        pred_labels,
        gt_labels,
        return_results=False,
        background_label=None,
        matching_score='overlap',
        matching_threshold=1):

    voxel_size = gt_labels.spec.voxel_size

    pred_labels_data = pred_labels.data
    pred_labels_spec = pred_labels.spec.copy()
    gt_labels_data = gt_labels.data
    gt_labels_spec = gt_labels.spec.copy()

    # PIXEL-WISE SCORES

    sample_scores = {
        'ppv': 0,
        'tpr': 0,
        'tnr': 0,
        'fpr': 0,
        'fscore': 0,
        'balanced_accuracy': 0
    }

    if background_label is not None:
        sample_scores.update({
            'ppv_fg': 0,
            'tpr_fg': 0,
            'tnr_fg': 0,
            'fpr_fg': 0,
            'fscore_fg': 0,
            'balanced_accuracy_fg': 0
        })

    # accuracy

    sample_scores['accuracy'] = (pred_labels_data == gt_labels_data).sum()/gt_labels_data.size
    if background_label is not None:
        fg_mask = gt_labels_data != background_label
        sample_scores['fg_accuracy'] = (
            pred_labels_data[fg_mask] ==
            gt_labels_data[fg_mask]).sum()/fg_mask.sum()

    # precision, recall, fscore

    label_ids = np.unique(gt_labels_data).astype(np.int32)
    for label in label_ids:

        relevant = gt_labels_data == label
        selected = pred_labels_data == label
        num_relevant = relevant.sum()
        num_selected = selected.sum()

        tp = (pred_labels_data[relevant] == label).sum()
        fp = num_selected - tp
        tn = (gt_labels_data.size - num_relevant) - fp

        # precision, or positive predictive value
        if num_selected > 0:
            ppv = tp/num_selected  # = tp/(tp + fp)
        else:
            ppv = np.nan
        # recall, or true positive rate
        if num_relevant > 0:
            tpr = tp/num_relevant  # = tp/(tp + fn)
        else:
            tpr = np.nan
        # specificity, or true negative rate
        if tn + fp > 0:
            tnr = tn/(tn + fp)
        else:
            tnr = np.nan
        # fall-out, or false positive rate
        if tn + fp > 0:
            fpr = fp/(tn + fp)
        else:
            fpr = np.nan

        if ppv + tpr > 0:
            fscore = 2*(ppv*tpr)/(ppv + tpr)
        else:
            fscore = np.nan
        balanced_accuracy = (tpr + tnr)/2

        sample_scores[f'ppv_{label}'] = ppv
        sample_scores[f'tpr_{label}'] = tpr
        sample_scores[f'tnr_{label}'] = tnr
        sample_scores[f'fpr_{label}'] = fpr
        sample_scores[f'fscore_{label}'] = fscore
        sample_scores[f'balanced_accuracy_{label}'] = balanced_accuracy
        sample_scores[f'ppv'] += ppv
        sample_scores[f'tpr'] += tpr
        sample_scores[f'tnr'] += tnr
        sample_scores[f'fpr'] += fpr
        sample_scores[f'fscore'] += fscore
        sample_scores[f'balanced_accuracy'] += balanced_accuracy
        if background_label is not None and label != background_label:
            sample_scores[f'ppv_fg'] += ppv
            sample_scores[f'tpr_fg'] += tpr
            sample_scores[f'tnr_fg'] += tnr
            sample_scores[f'fpr_fg'] += fpr
            sample_scores[f'fscore_fg'] += fscore
            sample_scores[f'balanced_accuracy_fg'] += balanced_accuracy

    num_classes = label_ids.size
    sample_scores[f'ppv'] /= num_classes
    sample_scores[f'tpr'] /= num_classes
    sample_scores[f'tnr'] /= num_classes
    sample_scores[f'fpr'] /= num_classes
    sample_scores[f'fscore'] /= num_classes
    sample_scores[f'balanced_accuracy'] /= num_classes
    if background_label is not None and num_classes >= 2:
        sample_scores[f'ppv_fg'] /= num_classes - 1
        sample_scores[f'tpr_fg'] /= num_classes - 1
        sample_scores[f'tnr_fg'] /= num_classes - 1
        sample_scores[f'fpr_fg'] /= num_classes - 1
        sample_scores[f'fscore_fg'] /= num_classes - 1
        sample_scores[f'balanced_accuracy_fg'] /= num_classes - 1

    # DETECTION SCORES (on foreground objects only)

    # limit detection scores to foreground labels
    if background_label is not None:
        label_ids = label_ids[label_ids != background_label]
    detection_scores = funlib.evaluate.detection_scores(
        gt_labels_data,
        pred_labels_data,
        label_ids,
        matching_score,
        matching_threshold,
        voxel_size=voxel_size,
        return_matches=return_results)
    for k, v in detection_scores.items():
        sample_scores[f'detection_{k}'] = v
    non_score_keys = []
    for k in sample_scores.keys():
        if k.startswith('detection_components_'):
            non_score_keys.append(k)
        elif k.startswith('detection_matches_'):
            non_score_keys.append(k)
    for k in non_score_keys:
        del sample_scores[k]

    sample_scores['detection_ppv'] = 0.0
    sample_scores['detection_tpr'] = 0.0
    sample_scores['detection_fscore'] = 0.0
    components = {}
    for label in label_ids:

        tp = detection_scores[f'tp_{label}']
        fp = detection_scores[f'fp_{label}']
        fn = detection_scores[f'fn_{label}']
        num_selected = tp + fp
        num_relevant = tp + fn

        # precision, or positive predictive value
        if num_selected > 0:
            ppv = tp/num_selected  # = tp/(tp + fp)
        else:
            ppv = np.nan
        # recall, or true positive rate
        if num_relevant > 0:
            tpr = tp/num_relevant  # = tp/(tp + fn)
        else:
            tpr = np.nan

        if ppv + tpr > 0:
            fscore = 2*(ppv*tpr)/(ppv + tpr)
        else:
            fscore = np.nan

        sample_scores[f'detection_ppv_{label}'] = ppv
        sample_scores[f'detection_tpr_{label}'] = tpr
        sample_scores[f'detection_fscore_{label}'] = fscore

        sample_scores['detection_ppv'] += ppv
        sample_scores['detection_tpr'] += tpr
        sample_scores['detection_fscore'] += fscore

        if return_results:

            components_gt = detection_scores[f'components_truth_{label}']
            components_pred = detection_scores[f'components_test_{label}']
            matches = detection_scores[f'matches_{label}']
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

            components_spec = gt_labels_spec.copy()
            components_spec.dtype = components_tp_gt.dtype

            components[f"components_tp_gt_{label}"] = gp.Array(
                components_tp_gt, components_spec.copy()
            )
            components[f"components_fn_gt_{label}"] = gp.Array(
                components_fn_gt, components_spec.copy()
            )
            components[f"components_tp_pred_{label}"] = gp.Array(
                components_tp_pred, components_spec.copy()
            )
            components[f"components_fp_pred_{label}"] = gp.Array(
                components_fp_pred, components_spec.copy()
            )
            
    num_classes = label_ids.size
    if num_classes >= 1:
        sample_scores['detection_ppv'] /= num_classes
        sample_scores['detection_tpr'] /= num_classes
        sample_scores['detection_fscore'] /= num_classes

    scores = {'sample': sample_scores, 'average': sample_scores}

    if return_results:
        pred_labels_spec.dtype = np.uint64
        gt_labels_spec.dtype = np.uint64
        results = {
            "pred_labels": gp.Array(
                pred_labels_data.astype(np.uint64), pred_labels_spec
            ),
            "gt_labels": gp.Array(gt_labels_data.astype(np.uint64), gt_labels_spec),
        }
        for k, v in components.items():
            spec = v.spec
            spec.dtype = np.uint64
            v_uint64 = gp.Array(v.data.astype(np.uint64), spec)
            results[k] = v_uint64

        return scores, results

    return scores
