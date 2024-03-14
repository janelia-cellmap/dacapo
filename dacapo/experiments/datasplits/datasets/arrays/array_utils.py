from funlib.persistence import open_ds
import zarr
import numpy as np

def open_dataset(f, ds):
    original_ds = ds
    slices_str = original_ds[len(ds):]

    try:
        dataset_as = []
        if all(key.startswith("s") for key in zarr.open(f)[ds].keys()):
            raise AttributeError("This group is a multiscale array!")
        for key in zarr.open(f)[ds].keys():
            dataset_as.extend(open_dataset(f, f"{ds}/{key}{slices_str}"))
        return dataset_as
    except AttributeError as e:
        # dataset is an array, not a group
        pass

    print("ds    :", ds)
    try:
        zarr.open(f)[ds].keys()
        is_multiscale = True
    except:
        is_multiscale = False

    if not is_multiscale:
        a = open_ds(f, ds)

        if a.data.dtype == np.int64 or a.data.dtype == np.int16:
            print("Converting dtype in memory...")
            a.data = a.data[:].astype(np.uint64)

        return [(a, ds)]
    else:
        return [([open_ds(f, f"{ds}/{key}") for key in zarr.open(f)[ds].keys()], ds)]