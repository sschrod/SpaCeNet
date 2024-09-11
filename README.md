# SpaCeNet

This package implements *SpaCeNet*. It is an extension of the traditional Gaussian Graphical Model which allows modelling of spatially distributed observations and the associations of their variables.

## Setup
You can either install the package directly from github via

```
pip install git+https://github.com/sschrod/SpaCeNet.git
```

or use the provided `Docker` image
```shell
docker build -t spacenet -f DOCKERFILE .
```
SpaCeNet is implemented using PyTorch Tensors to facilitate efficient matrix computations on the GPU. Hence, we suggest setting up PyTorch with CUDA.


## Get started
For a quick introduction we prepared a [jupyter notebook](https://github.com/sschrod/SpaCeNet/blob/master/example/quick_start.ipynb) demonstrating the inference of spatial relationships on simulated data with SpaCeNet.

The simplest way to analyse your own data using SpaCeNet is with `SpaCeNet_main.py`. Simply save the preprocessed data in a compressed numpy format
```python
np.savez(<file path>, X_mat=<Gene matrix (Sxnxp)>, coord_mat=<associated coordinates (Nxp)>, GeneNames=<List of Gene names (optional)>)
```
and call `SpaCeNet_main.py` with `--exp_name`, `--data_path`, `--preprocessed_data`, `--results_folder`. Further, set `-gs True` to run a grid-search, `-sr True` to run a single SpaCeNet model or `-ar True` for some general analysis.
For the full list of argparse arguments refer to `SpaCeNet_main.py`

To run a hyper-parameter search and analyse the findings on the simulated data run
```shell
python3 SpaCeNet_main.py --exp_name Simulation --data_path example --preprocessed_data simulated_data.pickle --results_folder sim_results/ -ss 1e-6 -e 1e-5 -l 3 -gs True -ar True
```

To reproduce the results on the MOSTA data, download the data (instructions in `\data`), run ```preprocess_MOSTA.py``` to save the preprocessed data and call
```shell
python3 SpaCeNet_main.py --exp_name MOSTA30 --data_path data --preprocessed_data MouseBrainAdult_30Percent.npz --results_folder results/ -nr 2 -st True -gs True -ar True```
```

## References
Schrod, Stefan, et al. "Spatial Cellular Networks from omics data with SpaCeNet." Genome Research (2024): gr-279125.

Schrod, Stefan, et al. "Spacenet: spatial cellular networks from omics data." International Conference on Research in Computational Molecular Biology. Cham: Springer Nature Switzerland, 2024.



