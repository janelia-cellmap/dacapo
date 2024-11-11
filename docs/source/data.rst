.. _sec_data:

Data Formatting
===============

Overview
--------

We support any data format that can be opened with the `zarr.open` convenience function from
`zarr <https://zarr.readthedocs.io/en/stable/api/convenience.html#zarr.convenience.open>`_. We also expect some specific metadata to come
with the data.

Metadata
--------

- `voxel_size`: The size of each voxel in the dataset. This is expected to be a tuple of ints
  with the same length as the number of spatial dimensions in the dataset.
- `offset`: The offset of the dataset. This is expected to be a tuple of ints with the same length
  as the number of spatial dimensions in the dataset.
- `axis_names`: The name of each axis. This is expected to be a tuple of strings with the same length
  as the total number of dimensions in the dataset. For example a 3D dataset with channels would have
  `axis_names=('c^', 'z', 'y', 'x')`. Note we expect non-spatial dimensions to include a "^" character.
  See [1]_ for expected future changes
- `units`: The units of each axis. This is expected to be a tuple of strings with the same length
  as the number of spatial dimensions in the dataset. For example a 3D dataset with channels would have
  `units=('nanometers', 'nanometers', 'nanometers')`.

Orgnaization
------------

Ideally all of your data will be contained in a single zarr container.
The simplest possible dataset would look like this:
::

    data.zarr
    ├── train
    │   ├── raw
    │   └── labels
    └── test
        ├── raw
        └── labels

If this is what your data looks like, then your data configuration will look like this:
.. code-block::
    :caption: A simple data configuration

    data_config = DataConfig(
        path="/path/to/data.zarr"
    )

Note that a lot of assumptions will be made.
1. We assume your raw data is normalized based on the `dtype`. I.e.
  if your data is stored as an unsigned int (we recommend uint8) we will assume a range and normalize it to [0,1] by dividing by the
  appropriate value (255 for `uint8` or 65535 for `uint16`). If your data is stored as any `float` we will assume
  it is already in the range [0, 1].
2. We assume your labels are stored as unsigned integers. If you want to generate instance segmentations, you will need
  to assign a unique id to every object of the class you are interested in. If you want semantic segmentations you
  can simply assign a unique id to each class. 0 is reserved for the background class.
3. We assume that the labels are provided densely. The entire volume will be used for training.
  

This will often not be enough to describe your data. You may have multiple crops and often your data may be
sparsely annotated. The same data configuration from above will also work for the slightly more complicated
dataset below:

::

    data.zarr
    ├── train
    │   ├── crop_01
    │   │   ├── raw
    │   │   ├── labels
    │   │   └── mask
    │   └── crop_02
    │       ├── raw
    │       └── labels
    └── test
        └─ crop_03
        │   ├── raw
        │   ├── labels
        │   └── mask
        └─ crop_04
            ├── raw
            └── labels

Note that `crop_01` and `crop_03` have masks associated with them. We assume a value of `0` in the mask indicates
unknown data. We will never use this data for supervised training, regardless of the corresponding label value.
If multiple test datasets are provided, this will increase the amount of information to review after training.
You will have e.g. `crop_03_voi` and `crop_04_voi` stored in the validation scores. Since we also take care to
save the "best" model checkpoint, you may now double the number of checkpoints saved since the checkpoint that
achieves optimal `voi` on `crop_03` may not be the same as the checkpoint that achieves optimal `voi` on `crop_04`.

Footnotes
---------

.. [1] The specification of axis names is expected to change in the future since we expect to support a `type` field in the future which
    can be one of ["time", "space", "{anything-else}"]. Which would allow you to specify dimensions as "channel"
    or "batch" or whatever else you want. This will bring us more in line with OME-Zarr and allow us to more easily
    handle a larger variety of common data specification formats.