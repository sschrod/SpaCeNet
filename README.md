# SpaCeNet

This package implements *SpaCeNet*. It is an extension of the traditional Gaussian Graphical Model which allows modelling of spatially distributed observations and the associations of their variables.

## Setup
You can either install the package directly from github via

```
pip install git+https://gitlab.gwdg.de/MedBioinf/NetworkInference/SpaCeNet.git
```

or you can download the repository and run `pip install .` in its root directory.

## Usage
The API is inspired by [scikit-learn](https://scikit-learn.org/) to achieve better usability. However, we use [pytorch](https://pytorch.org/)s  `torch.Tensor`s as data objects instead of `numpy.ndarray`s. This has the benefit that we can use cuda to speed up calculations.

All function and class functionality is documented within the respective docstring.

A jupyter notebook with a small demonstration is included in the `example` folder.