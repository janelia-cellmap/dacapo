<img src="docs/source/_static/CellMapLogo.png" alt="CellMap logo" width="85%">

# DaCapo ![DaCapo](docs/source/_static/icon_dacapo.png)

[![tests](https://github.com/janelia-cellmap/dacapo/actions/workflows/tests.yaml/badge.svg)](https://github.com/janelia-cellmap/dacapo/actions/workflows/tests.yaml)
[![black](https://github.com/janelia-cellmap/dacapo/actions/workflows/black.yaml/badge.svg)](https://github.com/janelia-cellmap/dacapo/actions/workflows/black.yaml)
[![mypy](https://github.com/janelia-cellmap/dacapo/actions/workflows/mypy.yaml/badge.svg)](https://github.com/janelia-cellmap/dacapo/actions/workflows/mypy.yaml)
[![docs](https://github.com/janelia-cellmap/dacapo/actions/workflows/docs.yaml/badge.svg)](https://janelia-cellmap.github.io/dacapo/)
[![codecov](https://codecov.io/gh/janelia-cellmap/dacapo/branch/main/graph/badge.svg)](https://codecov.io/gh/janelia-cellmap/dacapo)

A framework for easy application of established machine learning techniques on large, multi-dimensional images.

`dacapo` allows you to configure machine learning jobs as combinations of
[DataSplits](http://docs/api.html#datasplits),
[Architectures](http://docs/api.html#architectures),
[Tasks](http://docs/api.html#tasks),
[Trainers](http://docs/api.html#trainers),
on arbitrarily large volumes of
multi-dimensional images. `dacapo` is not tied to a particular learning
framework, but currently only supports [`torch`](https://pytorch.org/) with
plans to support [`tensorflow`](https://www.tensorflow.org/).

## Installation and Setup
Currently, python>=3.10 is supported. We recommend creating a new conda environment for dacapo with python 3.10.
```
conda create -n dacapo python=3.10
```

Then clone this repository, go into the directory, and install:
```
git clone git@github.com:janelia-cellmap/dacapo.git
cd dacapo
pip install .
```
This will install the minimum required dependencies. 

You may additionally utilize a MongoDB server for storing outputs. To install and run MongoDB locally, refer to the MongoDB documentation [here](https://www.mongodb.com/docs/manual/installation/).

The use of MongoDB, as well as specifying the compute context (on cluster or not) should be specified in the ```dacapo.yaml``` in the main directory.

## Functionality Overview

Tasks we support and approaches for those tasks:
 - Instance Segmentation
    - Affinities
    - Local Shape Descriptors
 - Semantic segmentation
    - Signed distances
    - One-hot encoding of different types of objects
