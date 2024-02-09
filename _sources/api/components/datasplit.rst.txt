
DataSplit Reference
===================

.. automodule:: dacapo
   :noindex:

DataSplit
^^^^^^^^^

:code:`DataSplits` are collections of multiple :code:`DataSets`, with each
:code:`DataSet` assigned to a specific role. i.e. training data, validation data,
testing data, etc.

.. autoclass:: dacapo.experiments.datasplits.DataSplit
    :members:

.. autoclass:: dacapo.experiments.datasplits.TrainValidateDataSplit
    :members:

    Configured with :class:`dacapo.datasplits.datasplits.TrainValidateDataSplitConfig`
    
    


DataSet
^^^^^^^

:code:`DataSets` define a spatial region containing the necessary data
for training provided as multiple `Arrays`. This can include as much as
raw, ground_truth, and a mask, or it could be just raw data in the case
of self supervised models.

ABC:

.. autoclass:: dacapo.experiments.datasplits.datasets.Dataset
    :members:

Implementations:

.. autoclass:: dacapo.experiments.datasplits.datasets.RawGTDataset
    :members:

    Configured with :class:`dacapo.experiments.datasplits.datasets.RawGTDatasetConfig`

Arrays
^^^^^^

:code:`Arrays` define the interface for a contiguous spatial region of
data. This data can be raw, ground truth, a mask, or any other spatial
data. :code:`Arrays` can be a direct interface to some storage i.e. a
zarr/n5 container, tiff stack, or other data storage, or can be a
wrapper modifying another array. This might include operations such
as normalizing intensities for raw data, binarizing labels to generate
a mask, or upsampling and downsampling. Providing these operations as
wrappers around allows us to lazily fetch and transform the data we
need consistently in different contexts such as training or validation.

ABC:

.. autoclass:: dacapo.experiments.datasplits.datasets.arrays.Array
    :members:

Implementations:

.. autoclass:: dacapo.experiments.datasplits.datasets.arrays.ZarrArray
    :members:

    Configured with :class:`dacapo.experiments.datasplits.datasets.arrays.ZarrArrayConfig`

.. autoclass:: dacapo.experiments.datasplits.datasets.arrays.BinarizeArray
    :members:

    Configured with :class:`dacapo.experiments.datasplits.datasets.arrays.BinarizeArrayConfig`

.. autoclass:: dacapo.experiments.datasplits.datasets.arrays.ResampledArray
    :members:

    Configured with :class:`dacapo.experiments.datasplits.datasets.arrays.ResampledArrayConfig`

.. autoclass:: dacapo.experiments.datasplits.datasets.arrays.IntensitiesArray
    :members:

    Configured with :class:`dacapo.experiments.datasplits.datasets.arrays.IntensitiesArrayConfig`

.. autoclass:: dacapo.experiments.datasplits.datasets.arrays.MissingAnnotationsMask
    :members:

    Configured with :class:`dacapo.experiments.datasplits.datasets.arrays.MissintAnnotationsMaskConfig`

.. autoclass:: dacapo.experiments.datasplits.datasets.arrays.OnesArray
    :members:

    Configured with :class:`dacapo.experiments.datasplits.datasets.arrays.OnesArrayConfig`

.. autoclass:: dacapo.experiments.datasplits.datasets.arrays.ConcatArray
    :members:

    Configured with :class:`dacapo.experiments.datasplits.datasets.arrays.ConcatArrayConfig`

.. autoclass:: dacapo.experiments.datasplits.datasets.arrays.LogicalOrArray
    :members:

    Configured with :class:`dacapo.experiments.datasplits.datasets.arrays.LogicalOrArrayConfig`

.. autoclass:: dacapo.experiments.datasplits.datasets.arrays.CropArray
    :members:

    Configured with :class:`dacapo.experiments.datasplits.datasets.arrays.CropArrayConfig`
