<img src="https://raw.githubusercontent.com/janelia-cellmap/dacapo/main/docs/source/_static/CellMapLogo.png" alt="CellMap logo" width="85%">

# DaCapo ![DaCapo](https://raw.githubusercontent.com/janelia-cellmap/dacapo/main/docs/source/_static/icon_dacapo.png) ![GitHub Org's stars](https://img.shields.io/github/stars/Janelia-cellmap/dacapo)

[![Documentation Status](https://readthedocs.org/projects/dacapo/badge/?version=stable)](https://dacapo.readthedocs.io/en/stable/?badge=stable)
![Github Created At](https://img.shields.io/github/created-at/funkelab/dacapo)
![GitHub License](https://img.shields.io/github/license/janelia-cellmap/dacapo)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fjanelia-cellmap%2Fdacapo%2Fmain%2Fpyproject.toml)

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

Then install DaCapo using pip with the following command:
```
pip install git+https://github.com/janelia-cellmap/dacapo
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
  
## Helpful Resources & Tools
 - Chunked data, zarr, and n5
    - OME-Zarr: a cloud-optimized bioimaging file format with international community support (doi: [10.1101/2023.02.17.528834](https://pubmed.ncbi.nlm.nih.gov/36865282/))
    - Videos about N5 and Fiji can be found in [this playlist](https://www.youtube.com/playlist?list=PLmZHHIZ9Gz-IJA7HtW8quZcuLViz9Em6e). For other questions, join the discussion on the [Image.sc forum](https://forum.image.sc/tag/n5).
    - Read about chunked storage plugins in Fiji in this blog: [N5 plugins for Fiji](https://openorganelle.janelia.org/news/2023-02-06-n5-plugins-for-fiji)
    - Script for converting tif to zarr can be found [here](https://github.com/yuriyzubov/tif-to-zarr) 
 - Segmentations
    - A description of local shape descriptors used for affinities task. Read the blog [here](https://localshapedescriptors.github.io/). Example image from the blog showing the difference between segmentations:
    - ![](https://localshapedescriptors.github.io/assets/img/detection_vs_segmentation_neurons.jpeg)
 - CellMap Models
    - [GitHub Repo](https://github.com/janelia-cellmap/cellmap-models) of published models
    - For example, the COSEM trained pytorch networks are located [here](https://github.com/janelia-cellmap/cellmap-models/tree/main/src/cellmap_models/pytorch/cosem).
 - [OpenOrganelle.org](https://openorganelle.janelia.org)
    - ![](https://raw.githubusercontent.com/janelia-cellmap/dacapo/main/docs/source/_static/mito_pred-seg.gif)
    - Example of [unprocessed distance predictions](https://tinyurl.com/3kw2tuab)
    - Example of [refined segmentations](https://tinyurl.com/k59pba98) that have undergone post-processing (e.g., thresholding, masking, smoothing)
    - Example of [groundtruth data](https://tinyurl.com/pu8mespz)
 - Visualization
    - [Neuroglancer GitHub Repo](https://github.com/google/neuroglancer)
   
