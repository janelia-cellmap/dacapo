import funlib.evaluate
import numpy as np
import zarr


def evaluate_labels(
        logits,
        gt_labels,
        store_results,
        background_label=None,
        matching_score='overlap',
        matching_threshold=1):

    pred_labels = np.argmax(logits.data, axis=0)
    gt_labels = gt_labels.data

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

    sample_scores['accuracy'] = (pred_labels == gt_labels).sum()/gt_labels.size
    if background_label is not None:
        fg_mask = gt_labels != background_label
        sample_scores['fg_accuracy'] = (
            pred_labels[fg_mask] ==
            gt_labels[fg_mask]).sum()/fg_mask.sum()

    # precision, recall, fscore

    label_ids = np.unique(gt_labels).astype(np.int32)
    for label in label_ids:

        relevant = gt_labels == label
        selected = pred_labels == label
        num_relevant = relevant.sum()
        num_selected = selected.sum()

        tp = (pred_labels[relevant] == label).sum()
        fp = num_selected - tp
        tn = (gt_labels.size - num_relevant) - fp

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
        gt_labels,
        pred_labels,
        label_ids,
        matching_score,
        matching_threshold,
        voxel_size=logits.spec.voxel_size)
    for k, v in detection_scores.items():
        sample_scores[f'detection_{k}'] = v

    sample_scores['detection_ppv'] = 0.0
    sample_scores['detection_tpr'] = 0.0
    sample_scores['detection_fscore'] = 0.0
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

    num_classes = label_ids.size
    if num_classes >= 1:
        sample_scores['detection_ppv'] /= num_classes
        sample_scores['detection_tpr'] /= num_classes
        sample_scores['detection_fscore'] /= num_classes

    if store_results:

        f = zarr.open(store_results)
        f['pred_labels'] = pred_labels.astype(np.uint64)
        f['gt_labels'] = gt_labels.astype(np.uint64)

    return {'sample': sample_scores}
