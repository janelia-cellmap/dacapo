from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import distance_transform_edt
from tqdm import tqdm
import mahotas
import numpy as np
import waterz
import zarr


def watershed_from_affinities(
        affs,
        max_affinity_value=1.0,
        fragments_in_xy=False,
        return_seeds=False,
        min_seed_distance=10):
    '''Extract initial fragments from affinities using a watershed
    transform. Returns the fragments and the maximal ID in it.
    Returns:
        (fragments, max_id)
        or
        (fragments, max_id, seeds) if return_seeds == True'''

    if fragments_in_xy:

        mean_affs = 0.5*(affs[1] + affs[2])
        depth = mean_affs.shape[0]

        fragments = np.zeros(mean_affs.shape, dtype=np.uint64)
        if return_seeds:
            seeds = np.zeros(mean_affs.shape, dtype=np.uint64)

        id_offset = 0
        for z in range(depth):

            boundary_mask = mean_affs[z]>0.5*max_affinity_value
            boundary_distances = distance_transform_edt(boundary_mask)

            ret = watershed_from_boundary_distance(
                boundary_distances,
                return_seeds=return_seeds,
                id_offset=id_offset,
                min_seed_distance=min_seed_distance)

            fragments[z] = ret[0]
            if return_seeds:
                seeds[z] = ret[2]

            id_offset = ret[1]

        ret = (fragments, id_offset)
        if return_seeds:
            ret += (seeds,)

    else:

        boundary_mask = np.mean(affs, axis=0)>0.5*max_affinity_value
        boundary_distances = distance_transform_edt(boundary_mask)

        ret = watershed_from_boundary_distance(
            boundary_distances,
            return_seeds,
            min_seed_distance=min_seed_distance)

        fragments = ret[0]

    return ret


def watershed_from_boundary_distance(
        boundary_distances,
        return_seeds=False,
        id_offset=0,
        min_seed_distance=10):

    max_filtered = maximum_filter(boundary_distances, min_seed_distance)
    maxima = max_filtered==boundary_distances
    seeds, n = mahotas.label(maxima)

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds!=0] += id_offset

    fragments = mahotas.cwatershed(
        boundary_distances.max() - boundary_distances,
        seeds)

    ret = (fragments.astype(np.uint64), n + id_offset)
    if return_seeds:
        ret = ret + (seeds.astype(np.uint64),)

    return ret


def evaluate_affs(affs, labels, dims, store_results=None):

    num_samples = affs.data.shape[0]
    scores = {}
    segmentations = []

    if dims == 2:

        # get all fragments at once for all samples
        # (use samples as z dimension)

        # (2, s, h, w)
        a = affs.data.transpose((1, 0, 2, 3))
        # (3, s, h, w)
        a = np.concatenate([np.zeros_like(a[0:1, :]), a])

        # (s, h, w)
        fragments, _ = watershed_from_affinities(a, fragments_in_xy=True)

    else:

        fragments = [
            watershed_from_affinities(affs[i])[0]
            for i in range(num_samples)
        ]

    for i in tqdm(range(num_samples), desc="evaluate"):

        a = affs.data[i].astype(np.float32)
        l = labels.data[i].astype(np.uint32)
        f = fragments[i].astype(np.uint64)

        if dims == 2:

            # convert to 3D
            a = np.concatenate(
                [np.zeros((1, 1) + a.shape[1:], dtype=np.float32),
                a[:,np.newaxis,:,:]])
            l = l[np.newaxis,:,:]
            f = f[np.newaxis,:,:]

        for segmentation, metrics in waterz.agglomerate(
                affs=a,
                thresholds=[0.5],
                gt=l,
                fragments=f):
            segmentations.append(segmentation)
            scores[f'sample_{i}'] = {
                'voi_split': metrics['V_Info_split'],
                'voi_merge': metrics['V_Info_merge'],
                'rand_split': metrics['V_Rand_split'],
                'rand_merge': metrics['V_Rand_merge']
            }

    if store_results:

        f = zarr.open(store_results)
        f['fragments'] = np.stack(fragments)
        f['segmentation'] = np.concatenate(segmentations)
        f['labels'] = labels.data

    return scores
