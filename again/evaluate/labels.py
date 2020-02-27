import numpy as np
import zarr


def evaluate_labels(logits, gt_labels, store_results, background_label=None):

    pred_labels = np.argmax(logits.data, axis=0)
    gt_labels = gt_labels.data

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
        ppv = tp/num_selected  # = tp/(tp + fp)
        # recall, or true positive rate
        tpr = tp/num_relevant  # = tp/(tp + fn)
        # specificity, or true negative rate
        tnr = tn/(tn + fp)
        # fall-out, or false positive rate
        fpr = fp/(tn + fp)

        fscore = 2*(ppv*tpr)/(ppv + tpr)
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
    if background_label is not None:
        sample_scores[f'ppv_fg'] /= num_classes - 1
        sample_scores[f'tpr_fg'] /= num_classes - 1
        sample_scores[f'tnr_fg'] /= num_classes - 1
        sample_scores[f'fpr_fg'] /= num_classes - 1
        sample_scores[f'fscore_fg'] /= num_classes - 1
        sample_scores[f'balanced_accuracy_fg'] /= num_classes - 1

    if store_results:

        f = zarr.open(store_results)
        f['pred_labels'] = pred_labels.astype(np.uint64)
        f['gt_labels'] = gt_labels.astype(np.uint64)

    return {'sample': sample_scores}
