# %%
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional, Tuple


# %%
def AUC(
        params_est: np.ndarray,
        params_true: np.ndarray,
        plot: bool = True,
        axes: Optional[Tuple[plt.Axes]] = None
) -> Tuple[float, float, np.ndarray]:
    r"""
    Calculates the AUC of ROC and precision-recall curve based on estimated and true
    parameters. Curves can be plotted, even into a tuple of custom `axes`.

    Specifically designed for $\Omega$ and $\Delta\rho$ in SpaCeNet. Given parameters
    must be 2- or 3-dimensional numpy arrays and symmetric with respect to switching
    of the last two axis, since only the upper-triangular values areconsidered. If
    3-dimensional arrays are provided, positions are labelled as positives if any value
    along the first axis exceeds the thresholding value.

    Parameters
    ----------
    params_est : numpy.ndarray
        A numpy array containing the estimated parameters.
    params_true : numpy.ndarray
        A numpy array containing the true parameters.
    plot : bool
        Whether or not ROC and precision-recall curves should be plotted.
    axes : Optional[Tuple[matplotlib.pyplot.Axes]]
        The user can specify where axes where the plots should be displayed. If `None`
        are provided a new figure will be created. Defaults to `None`.

    Returns
    -------
    tuple
        A tuple `(AUROC, AUPRC, measures)` containing the AUROC and AUPRC values as
        well as a numpy.ndarray with the columns true_positives, false_positives,
        true_negatives, false_negatives, false_positive_rate, true_positive_rate,
        recall and precision, where each row corresponds to one thresholding value.
    """

    # preprocess parameters
    p = params_est.shape[-1]
    n_triu = int(p * (p + 1) / 2)
    triu_indices_mat = np.triu_indices(p)

    if params_true.ndim == 2:
        params_true = np.expand_dims(params_true, 0)
        L_true = 1
    elif params_true.ndim == 3:
        L_true = params_true.shape[0]
    else:
        raise ValueError("bad dimensions params_true")
    triu_indices_true = (
        np.repeat(np.arange(L_true), n_triu),
        *np.tile(np.array(triu_indices_mat), L_true)
    )

    if params_est.ndim == 2:
        params_est = np.expand_dims(params_est, 0)
        L_est = 1
    elif params_est.ndim == 3:
        L_est = params_est.shape[0]
    else:
        raise ValueError("bad dimensions params_est")
    triu_indices_est = (
        np.repeat(np.arange(L_est), n_triu),
        *np.tile(np.array(triu_indices_mat), L_est)
    )

    params_true = params_true[triu_indices_true].reshape((n_triu, L_true), order="F")
    params_est = np.abs(params_est[triu_indices_est]).reshape((n_triu, L_est), order="F")

    # get true parameter metrics
    positives_true = np.any(params_true != 0.0, axis=1)
    n_true = np.sum(positives_true)
    n_false = np.sum(~ positives_true)

    # compare to estimated parameters based on thresholds
    thresholds = np.concatenate([
        [np.min(params_est) - 0.1],
        np.unique(params_est),
        [np.max(params_est) + 0.1]
    ])
    measures = np.zeros((len(thresholds), 8))
    for i, thresh in enumerate(thresholds):
        positives_est = np.any(params_est > thresh, axis=1)
        TP = np.sum(np.logical_and(positives_est, positives_true))
        FP = np.sum(np.logical_and(positives_est, ~ positives_true))
        TN = np.sum(np.logical_and(~ positives_est, ~ positives_true))
        FN = np.sum(np.logical_and(~ positives_est, positives_true))
        TPR = TP / n_true
        FPR = FP / n_false
        precision = 1 if TP == FP == 0 else TP / (TP + FP)
        recall = TP / (TP + FN)
        measures[i] = TP, FP, TN, FN, FPR, TPR, recall, precision

    # calculate AUC = sum(x_diff * y_diff / 2)
    AUROC = np.sum((measures[:-1, 4] - measures[1:, 4]) * (measures[:-1, 5] + measures[1:, 5]) / 2)
    AUPRC = np.sum((measures[:-1, 6] - measures[1:, 6]) * (measures[:-1, 7] + measures[1:, 7]) / 2)

    # plot if desired
    if plot:
        if axes is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)
        else:
            ax1, ax2 = axes
        if ax1 is not None:
            ax1.plot([0, 1], [0, 1], "--", c="grey")
            ax1.plot(measures[:, 4], measures[:, 5])
            ax1.set_xlim([-0.02, 1.02])
            ax1.set_ylim([-0.02, 1.02])
            ax1.set_aspect('equal', adjustable='box')
            ax1.set_xlabel("False Positive Rate")
            ax1.set_ylabel("True Positive Rate")
            ax1.set_title(f"AUROC = {AUROC:.3f}")
        if ax2 is not None:
            ax2.plot([0, 1], [0, 0], "--", c="grey")
            ax2.plot(measures[:, 6], measures[:, 7])
            ax2.set_xlim([-0.02, 1.02])
            ax2.set_ylim([-0.02, 1.02])
            ax2.set_aspect('equal', adjustable='box')
            ax2.set_xlabel("Recall")
            ax2.set_ylabel("Precision")
            ax2.set_title(f"AUPRC = {AUPRC:.3f}")

    return AUROC, AUPRC, measures


def cross_dist(a_coords: np.ndarray, b_coords: np.ndarray, method: str = "euclidean") -> np.array:
    r"""Calculate the pairwise distances between two 2d arrays with matching numbers of columns.

    Parameters
    ----------
    a_coords : numpy.ndarray
        A numpy array with shape `[n, d]`.
    b_coords : numpy.ndarray
        A numpy array with shape `[m, d]`.
    method : str
        How to compute the distance along the `d` dimension. Options are "euclidean",
        "squared_euclidean" and "manhattan". Default is "euclidean".

    Returns
    -------
    numpy.ndarray
        A numpy array with shape `[n, m]` containing the cross distances between the
        input arrays.
    """

    if method == "euclidean":
        return np.sqrt(cross_dist(a_coords, b_coords, "squared_euclidean"))

    diff_tens = a_coords[:, None, :] - b_coords[None, :, :]
    if method == "squared_euclidean":
        dist_mat = np.sum(diff_tens ** 2, axis=2)
    elif method == "manhattan":
        dist_mat = np.sum(np.abs(diff_tens), axis=2)
    else:
        raise NotImplementedError(f"method '{method}' not implemented")
    return dist_mat


def get_device(use_cuda: bool) -> torch.device:
    r"""Get a `torch.device` based on preference and system availablity.

    Parameters
    ----------
    use_cuda : bool
        Whether or not to use cuda.

    Returns
    -------
    torch.device
        Returns a `torch.device`. If cuda is desired but not available, CPU is used.
    """

    if use_cuda and not torch.cuda.is_available():
        print_colored("Error: cuda is not available, default to cpu!", "red")
        device = torch.device("cpu")
    elif not use_cuda:
        print_colored("Info: will use cpu!", "blue")
        device = torch.device("cpu")
    else:
        print_colored("Info: cuda requested and available, will use gpu!", "green")
        device = torch.device("cuda")
    return device


def pair_dist(coords: np.ndarray, method: str = "euclidean") -> np.ndarray:
    r"""Calculate the pairwise distances between all observations a 2d array.

    Parameters
    ----------
    coords : numpy.ndarray
        A numpy array with shape `[n, d]`.
    method : str
        How to compute the distance along the `d` dimension. Options are "euclidean",
        "squared_euclidean" and "manhattan". Default is "euclidean".

    Returns
    -------
    numpy.ndarray
        A numpy array with shape `[n, n]` containing the pairwise distances between the
        observations in the input array.
    """

    return cross_dist(coords, coords, method)


def print_colored(message: str, color: str) -> None:
    r"""Prints colored text.

    Parameters
    ----------
    message : str
        The message to be printed.
    color : str
        The desired color. Options for `color` are "red", "green", "yellow" and "blue".
    """
    if color == "red":
        print("\033[91m" + str(message) + "\033[0m")
    elif color == "green":
        print("\033[92m" + str(message) + "\033[0m")
    elif color == "yellow":
        print("\033[93m" + str(message) + "\033[0m")
    elif color == "blue":
        print("\033[94m" + str(message) + "\033[0m")
    else:
        raise ValueError("Invalid color selected.")


class StdoutSplitter:
    """Redirects the `stdout` stream to an `StringIO` instance and a potential
    `second_target`."""

    def __init__(self, second_target=None, **kwargs):
        self.stringio = StringIO(**kwargs)
        self.second_target = second_target

    def write(self, stream):
        self.stringio.write(stream)
        if self.second_target is not None:
            self.second_target.write(stream)

    def getvalue(self):
        return self.stringio.getvalue()


def Theta(
        coords: np.ndarray,
        base: str = "smoothed_potential",
        L: int = 1,
        scale_correction: Optional[str] = "min",
        standardize=False,
        dist_method: str = "euclidean",
        k: Optional[int] = None,
        quantile: Optional[float] = None
) -> np.ndarray:
    r"""Calculate the distance based expansions for SpaCeNet.

    Parameters
    ----------
    coords : numpy.ndarray
        A numpy array with shape `[n, d]` or `[S, n, d]` containing the coordinates of
        the observations. If a 2d array is provided, it is assumed that `S = 1` and a
        corresponding singleton dimension is inserted into the result.
    base : str
        In which base the expansion is performed. If 'Coulomb' is selected, the
        expansion is performed in $(r_0/r_{ab})^l$ and if 'smoothed_potential' is selected,
        the expasion is performed in $(r_0/r_{ab})^l * (1-exp[-r_{ab}/r_0])^l$ where
        $r_{ab}$ are the observed pairwise distances, $l = 1,...,L$ is the expansion order.
        Default is "smoothed_potential".
    L : int
        The order of the expansion. Default is 1.
    scale_correction : str, optional
        How to determine $r_0$. Options are "min", "mean", "meadian", "knn" or `None`.
        "knn" will calculate $r_0$ as a certain quantile of the nearest neighbor
        distances (see parameters `k` and `quantile`). `None` results in $r_0 = 1$ and no
        scale correction. Defaults to "min".
    standardize : boolean
        Whether or not the values of each expansion order are divided by their respective
        standard deviation. Default is False.
    dist_method : str
        How to compute the distance along the `d` dimension. Options are "euclidean",
        "squared_euclidean" and "manhattan". Default is "euclidean".
    k : int, optional
        How many nearest neighbors are considered if `knn` is chosen as `scale_correction`.
        It will be ignored for all other `scale_correction` values.
    quantile : float, optional
        Which quantile between 0 and 1 from the knn distances will be used as $r_0$.


    Returns
    -------
    numpy.ndarray
        A numpy array with shape `[S, L, n, n]` containing the $\Theta$-values.
    """

    if coords.ndim == 2:
        coords = coords[None, ...]
    S, n, d = coords.shape
    dist_tens = np.zeros((S, n, n))
    for s in range(S):
        dist_mat_s = pair_dist(coords[s], dist_method)
        np.fill_diagonal(dist_mat_s, float("nan"))  # ignore distance to same observation
        dist_tens[s] = dist_mat_s

    if scale_correction is None:
        r0 = 1
    elif scale_correction == "min":
        r0 = np.nanmin(dist_tens)
    elif scale_correction == "mean":
        r0 = np.nanmean(dist_tens)
    elif scale_correction == "median":
        r0 = np.nanmedian(dist_tens)
    elif (scale_correction == "knn"):
        if not isinstance(k, int):
            raise ValueError("Invalid 'k' for knn. Must be integer.")
        knn_dists = np.sort(dist_tens, axis=2)[:, :, :k]  # moves "nan" to end
        r0 = np.nanquantile(knn_dists, q=quantile)
    else:
        raise ValueError("Invalid 'scale_correction'!")
    r_inv = r0 / dist_tens

    if base == "Coulomb":
        pass
    elif base == "smoothed_potential":
        r_inv = r_inv * (1 - np.exp(-1 * r_inv ** (-1)))
    else:
        raise ValueError("Invalid expansion base. Choose 'Coulomb' or 'smoothed_potential'!")

    Theta_tens = r_inv[:, None, :, :] ** np.arange(1, L + 1)[None, :, None, None]

    if standardize:
        Theta_std = np.nanstd(Theta_tens, axis=(0, 2, 3))
        Theta_tens = Theta_tens / Theta_std[None, :, None, None]

    Theta_tens = np.nan_to_num(Theta_tens, nan=0.)  # set same obs entries to 0 again
    return Theta_tens


def unpickle(file, device):
    r"""
    Unpickles a *binary* pickled `file`. Any applicable `torch` object will be loaded
    to `device`. This allows to load cuda objects on a CPU only system.
    """

    import io
    import pickle
    from torch import load as torch_load

    class Custom_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda byte: torch_load(io.BytesIO(byte), map_location=device)
            else:
                return super().find_class(module, name)

    return Custom_Unpickler(file).load()
