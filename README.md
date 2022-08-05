![DaCapo](docs/source/_static/dacapo.svg)

![tests](https://github.com/pattonw/dacapo/actions/workflows/tests.yaml/badge.svg)
![black](https://github.com/pattonw/dacapo/actions/workflows/black.yaml/badge.svg)
![mypy](https://github.com/pattonw/dacapo/actions/workflows/mypy.yaml/badge.svg)
[![docs](https://readthedocs.org/projects/dacapo/badge/?version=latest&style=plastic)](https://dacapo.readthedocs.io/en/latest/)

A framework for easy application of establed machine learning techniques on large, multi-dimensional images.

`dacapo` allows you to configure machine learning jobs as combinations of
[DataSplits](http://docs/api.html#datasplits),
[Architectures](http://docs/api.html#architectures),
[Tasks](http://docs/api.html#tasks),
[Trainers](http://docs/api.html#trainers),
on arbitrarily large volumes of
multi-dimensional images. `dacapo` is not tied to a particular learning
framework, but currently only supports [`torch`](https://pytorch.org/) with
plans to support [`tensorflow`](https://www.tensorflow.org/).