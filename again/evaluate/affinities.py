from tqdm import tqdm
import numpy as np
import waterz
import zarr


def evaluate_affs(affs, labels, dims, store_results=None):

    num_samples = affs.data.shape[0]
    scores = {}
    segmentations = []

    for i in tqdm(range(num_samples), desc="evaluate"):

        if dims == 2:

            # (2, h, w)
            a = affs.data[i].astype(np.float32)
            # (h, w)
            l = labels.data[i].astype(np.uint32)

            # convert to 3D
            a = np.concatenate(
                [np.zeros((1, 1) + a.shape[1:], dtype=np.float32),
                a[:,np.newaxis,:,:]])
            l = l[np.newaxis,:,:]

        for segmentation, metrics in waterz.agglomerate(a, [0.5], l):
            segmentations.append(segmentation)
            scores[f'sample_{i}'] = metrics

    if store_results:

        f = zarr.open(store_results)
        f['segmentation'] = np.concatenate(segmentations)
        f['labels'] = labels.data

    return scores
