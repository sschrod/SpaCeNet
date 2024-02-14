# %%
from contextlib import redirect_stdout
import numpy as np
import pandas as pd
import sys
import torch
from tqdm import tqdm
import traceback as tb
from typing import List, Optional, Tuple, Union

from .optimizer import solve
from .utils import StdoutSplitter


# %%

class SpaCeNet:
    r"""An implementation of SpaCeNet.

    SpaCeNet fits a sparse inverse covariance matrix to spatially distributed data.
    It not only models associations of variables within observations but also
    between different observations.


    Attributes
    ----------
    alpha : float
        The hyperparameter used to penalize the within observation associations.
    beta : float
        The hyperparameter used to penalize the between observation associations.
    Drho_tens : torch.Tensor
        A $L \times p \times p$ tensor with the parameter estimations for $\Delta\rho$.
    final_eps : float
        The last relative change of losses between two iterations before termination of
        the algorithm.
    L : int
        The expansion order used in $\Theta$.
    losses : torch.Tensor
        All loss values per iteration.
    n : int
        The number of observations per sample.
    n_iter : int
        The number of iterations before termination.
    mu_vec : torch.Tensor
        A $p$-length vector with the estimated $\mu$ parameters, i.e. the location
        unspecific mean vector.
    Omega_mat : torch.Tensor
        A $p \times p$ matrix with the parameter estimations for $\Omega$, i.e. the
        within observation associations.
    outputs : str
        Saves the optimizer outputs if optimizer options include `verbose = True`.
    p : int
        The number of variables.
    S : int
        The number of samples.


    Methods
    -------
    fit(X_train, Theta_train, reset_params = False)
        Fits the model based on the training data `X_train` and `Theta_train`.
    information_criterion(which)
        Calculates AIC and/or BIC given the currently estimated parameters and
        training data.
    score(X_test, Theta_test)
        Calculates the pseudo-log-likelihood for test data.
    """

    def __init__(
            self,
            alpha: float = 0.,
            beta: float = 0.,
            penalize_diag: bool = False,
            device: torch.device = "cpu",
            optimizer_options: Optional[dict] = None
    ) -> None:
        r"""
        Parameters
        ----------
        alpha : float
            The hyperparameter used to penalize the within observation associations.
            Defaults to `0`.
        beta : float
            The hyperparameter used to penalize the between observation associations.
            Defaults to `0`.
        penalize_diag : bool
            Whether or not the diagonal of $\Omega$ (and therefore $\Lambda$) should be
            penalized as well. Defaults to `False`.
        device : torch.device
            The device where all calculations are performed. Must match the device of
            training and test data. Defaults to `"cpu"`.
        optimizer_options : dict, optional
            A dictionary with keywords and arguments that are passed on to the optimizer.
            Defaults to `None`.
        """

        self.alpha = alpha
        self.beta = beta
        self.losses = None
        self.final_eps = None
        self.penalize_diag = penalize_diag
        self.n_iter = None
        self.device = device
        self._params = None
        self.outputs = ""

        # set default optimizer options and overwrite with those given by user
        self.optimizer_options = dict(
            step_size=1E-6,
            momentum=0.9,
            eps=1e-5,
            verbose=True,
            max_iter=10_000,
            max_step_size_reductions=3,
            step_size_reduction_factor=10.,
        )
        if optimizer_options is not None:
            for key, value in optimizer_options.items():
                self.optimizer_options[key] = value

    def __repr__(self):
        return f"SpaCeNet(alpha={self.alpha!r}, beta={self.beta!r}, penalize_diag={self.penalize_diag!r}, device={self.device!r}, optimizer_options={self.optimizer_options!r})"

    @staticmethod
    def _check_dims(X_mat: torch.Tensor, Theta_tens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        An internal function to check the dimensions of $X$ & $\Theta$ and to insert
        missing dimensions where necessary.
        """
        assert isinstance(X_mat, torch.Tensor) and isinstance(Theta_tens, torch.Tensor)
        if X_mat.ndim == 2:
            # S = 1
            X_mat = torch.unsqueeze(X_mat, dim=0)
            if Theta_tens.ndim == 2:
                # L = 1, insert L & S dimensions
                Theta_tens = Theta_tens[None, None, ...]
            elif Theta_tens.ndim == 3:
                # assume L is provided, insert S dimension
                Theta_tens = torch.unsqueeze(Theta_tens, dim=0)
            elif Theta_tens.ndim == 4 and Theta_tens.shape[0] == 1:
                pass
            else:
                raise ValueError("Can't interpret the dimensions of Theta_tens!")
        elif X_mat.ndim == 3:
            # S != 1
            if Theta_tens.ndim == 3:
                # check if L or S are given
                if Theta_tens.shape[0] == X_mat.shape[0]:
                    # S matches, L = 1, insert L dimension
                    Theta_tens = torch.unsqueeze(Theta_tens, dim=1)
                else:
                    # S does not match, L is given --> insert S dimension
                    Theta_tens = torch.unsqueeze(Theta_tens, dim=0)
            elif Theta_tens.ndim == 2:
                # insert S & L dimensions
                Theta_tens = Theta_tens[None, None, ...]
            else:
                ValueError("Can't interpret the dimensions of Theta_tens!")
        else:
            raise ValueError("Can't interpret the dimensions of X_mat!")
        if not (X_mat.shape[1] == Theta_tens.shape[-1] == Theta_tens.shape[-2]):
            raise ValueError(
                f"Mismatch in observations of X_mat and Theta_tens: {X_mat.shape[1]} != {Theta_tens.shape[-2:]}")
        return X_mat, Theta_tens

    def fit(
            self,
            X_train: torch.Tensor,
            Theta_train: torch.Tensor,
            reset_params: bool = False
    ) -> None:
        r"""
        Fits the model based on the training data `X_train` and `Theta_train`.

        Data tensors should have data type `torch.float64`. If only one sample was
        observed ($S = 1$) or the expansion has order 1 ($L = 1$) some information
        may be inferred from low-dimensional tensors, but it is safer to provide the
        tensors with respective singleton dimensions as described below.

        Parameters
        ----------
        X_train : torch.Tensor
            A $S \times n \times p$ tensor with the observed data.
        Theta_train : torch.Tensor
            A $S \times L \times n \times n$ tensor with the spatial information.
        reset_params : bool
            Whether or not the parameters should be reset for repeated calls. May be
            useful if convergence has not yet been reached or similar data should be
            fitted. However, $L$ and $p$ must not change between the data sets.
            Defaults to `False`.
        """

        # prepare data tensors
        self.X_train, self.Theta_train = self._check_dims(X_train, Theta_train)
        self._X_mean = torch.mean(self.X_train, dim=(0, 1))
        self._X_std = torch.std(self.X_train, dim=(0, 1))
        self.X_train = (self.X_train - self._X_mean) / self._X_std
        self.S, self.n, self.p = self.X_train.shape
        self.L = self.Theta_train.shape[1]

        # prepare reusable indices
        self.n_triu = int(self.p * (self.p + 1) / 2)
        self.diag_indices_mat = np.diag_indices(self.p)
        self.triu_indices_mat = np.triu_indices(self.p, k=0)
        self.triu_indices_mat_offset = np.triu_indices(self.p, k=1)
        self.triu_indices_tens = (
            np.repeat(np.arange(self.L), self.n_triu),
            *np.tile(np.array(self.triu_indices_mat), self.L)
        )
        self.triu_indices_tens_offset = (
            np.repeat(np.arange(self.L), int(self.p * (self.p - 1) / 2)),
            *np.tile(np.array(self.triu_indices_mat_offset), self.L)
        )

        # get hyperparameters ready for proximal operator
        # Omega & Drho first
        hyper_weights = torch.repeat_interleave(
            self.n * self.S * torch.tensor([self.alpha, self.beta], device=self.device, dtype=torch.float64),
            torch.tensor([self.n_triu, self.n_triu], device=self.device, dtype=torch.int)
        )
        if not self.penalize_diag:  # of Omega (and therefore Lambda)
            diag_ind = torch.cat([
                torch.zeros(1, dtype=torch.int, device=self.device),
                torch.cumsum(torch.arange(self.p, 1, -1, device=self.device), 0)
            ])
            hyper_weights[:self.n_triu][diag_ind] = 0
        # add zeros for mu
        self.hyper_weights = torch.cat([
            torch.zeros(self.p, dtype=torch.float64, device=self.device),
            hyper_weights
        ])

        # allow warm start
        if reset_params or (self._params is None) or torch.any(torch.isnan(self._params)) or torch.all(
                self._params == 0):
            params_init = torch.zeros(self.p + self.n_triu * (1 + self.L), dtype=torch.float64, device=self.device)
            params_init[self.p:(self.p + self.n_triu)] = torch.eye(self.p)[self.triu_indices_mat].flatten()
        else:
            if len(self._params) != self.p + (1 + self.L) * self.n_triu:
                raise ValueError("L and p do not match previously fitted dataset. Please set `reset_params = True`.")
            params_init = self._params

        # fit the model
        with redirect_stdout(StdoutSplitter(sys.stdout if self.optimizer_options.get("verbose") else None)) as outputs:
            self._params, self.losses = solve(
                self._loss_grad,
                self._prox,
                params_init,
                **self.optimizer_options
            )
        self.outputs = outputs.getvalue()

        # retrieve fitting statistics
        self.n_iter = len(self.losses)
        self.final_eps = abs((self.losses[-1] - self.losses[-2]) / self.losses[-2])

    @property
    def mu_vec(self) -> torch.Tensor:
        if self._params is not None:
            return self._params[:self.p]
        else:
            return None

    @property
    def Omega_mat(self) -> torch.Tensor:
        if self._params is not None:
            return self._flat_to_symmetric(self._params[self.p:(self.p + self.n_triu)])
        else:
            return None

    @property
    def Drho_tens(self) -> torch.Tensor:
        if self._params is not None:
            Drho_params = self._params[(self.p + self.n_triu):]
            Drho_tens = torch.zeros((self.L, self.p, self.p), dtype=torch.float64, device=self.device)
            for l in range(self.L):
                Drho_tens[l] = self._flat_to_symmetric(
                    Drho_params[l * self.n_triu:(l + 1) * self.n_triu]
                )
            return Drho_tens
        else:
            return None

    def _flat_to_symmetric(self, params: torch.Tensor) -> torch.Tensor:
        r"""
        A function for internal use which transforms a flat parameter vector into a
        symmetric matrix.
        """

        matrix = torch.zeros((self.p, self.p), dtype=torch.float64, device=self.device)
        matrix[self.triu_indices_mat] = params
        matrix.T[self.triu_indices_mat_offset] = matrix[self.triu_indices_mat_offset]
        return matrix

    def _loss_grad(self, params: torch.Tensor) -> torch.Tensor:
        r"""
        A function for internal use which calculates loss and gradient at a given `params`
        position.
        """

        # assign to self to generate the right matrices
        self._params = params

        # calculate reusable parameters
        Omega_mat = self.Omega_mat
        Omega_diag = torch.diag(Omega_mat)
        XM_mat = self.X_train - self.mu_vec
        Drho_tens = self.Drho_tens
        Z_tens = self.Theta_train @ torch.unsqueeze(XM_mat, dim=1)
        Rho_mat = torch.tensordot(Z_tens, Drho_tens, dims=([1, 3], [0, 2]))
        A_mat = (Rho_mat + XM_mat @ Omega_mat) / Omega_diag

        """
        # Changed to speed up computation
        grad_mu = torch.tensordot(
          A_mat,
          Omega_mat + torch.tensordot(torch.sum(self.Theta_train, dim = -1), Drho_tens, dims = ([1], [0])), #xxx,   # [S, a, j, q]
          dims = ([0, 1, 2], [0, 1, 2])
         )"""

        grad_mu_2 = torch.sum(torch.tensordot(
            A_mat,
            Omega_mat,
            dims=([2], [0])
        ), (0, 1))

        grad_mu_3 = torch.tensordot(torch.sum(self.Theta_train, dim=-1), torch.tensordot(
            A_mat,
            Drho_tens,
            dims=([2], [1])
        ), ([0, 1, 2], [0, 2, 1]))

        grad_mu = grad_mu_2 + grad_mu_3

        grad_Drho = -1 * torch.sum(torch.unsqueeze(A_mat.swapaxes(1, 2), dim=1) @ Z_tens, dim=0)
        grad_Drho[self.triu_indices_tens_offset] += grad_Drho.swapaxes(1, 2)[self.triu_indices_tens_offset]
        grad_Omega = -1 * torch.sum(A_mat.swapaxes(1, 2) @ XM_mat, dim=0)
        grad_Omega[self.triu_indices_mat_offset] += grad_Omega.T[self.triu_indices_mat_offset]
        grad_Omega_diag = 0.5 * (
                self.n * self.S / Omega_diag - torch.sum(XM_mat ** 2, dim=(0, 1)) + torch.sum((A_mat - XM_mat) ** 2,
                                                                                              dim=(0, 1))
        )
        grad_Omega[self.diag_indices_mat] = grad_Omega_diag

        # flatten gradients
        gradient = -1 * torch.cat([
            grad_mu,
            grad_Omega[self.triu_indices_mat],
            grad_Drho[self.triu_indices_tens]
        ])

        pseudo_logLik = -0.5 * (
                self.S * self.n * self.p * np.log(2 * np.pi)
                - self.S * self.n * torch.sum(torch.log(Omega_diag))
                + torch.sum(A_mat ** 2 * Omega_diag)
        )
        return -1 * pseudo_logLik, gradient

    def _prox(self, params_vec: torch.Tensor, step_size: float) -> torch.Tensor:
        r"""
        A function for internal use which applies the proximal operator to the parameters
        for a given step size.
        """

        # proximal function of group lasso:
        # - Use the fact that group lasso reduces to normal lasso for Omega and mu to only
        #   call `norm` once (repeated calculation is slow).
        # - mu might actually be ignored here (unpenalized), but is kept in case penalization
        #   is desired at some point.
        params_mat = torch.zeros((self.p + 2 * self.n_triu, self.L), dtype=torch.float64, device=self.device)
        params_mat[:self.p, 0] = params_vec[:self.p]  # mu params
        params_mat[self.p:(self.p + self.n_triu), 0] = params_vec[self.p:(self.p + self.n_triu)]  # Omega params
        params_mat[(self.p + self.n_triu):, :] = params_vec[(self.p + self.n_triu):].reshape(
            (self.L, self.n_triu)).T  # Drho params

        norms = torch.linalg.norm(params_mat, dim=1)
        r = step_size * self.hyper_weights
        less = torch.less(r, norms)
        prox_mat = torch.zeros_like(params_mat, dtype=torch.float64, device=self.device)
        prox_mat[less] = (params_mat[less].T * (1 - r[less] / norms[less])).T

        prox_vec = torch.cat([
            prox_mat[:self.p, 0],
            prox_mat[self.p:(self.p + self.n_triu), 0],
            prox_mat[(self.p + self.n_triu):, :].T.flatten()
        ])
        return prox_vec

    def score(self, X_test: torch.Tensor, Theta_test: torch.Tensor) -> torch.Tensor:
        r"""
        Calculates the pseudo-log-likelihood for test data.

        Data tensors should have data type `torch.float64`. If only one sample was
        observed ($S = 1$) or the expansion has order 1 ($L = 1$) some information
        may be inferred from low-dimensional tensors, but it is safer to provide the
        tensors with respective singleton dimensions as described below.

        Parameters
        ----------
        X_test : torch.Tensor
            A $S \times n \times p$ tensor with the observed variable data.
        Theta_test : torch.Tensor
            A $S \times L \times n \times n$ tensor with the spatial information.

        Returns
        -------
        torch.Tensor
            A tensor with the calculated pseudo-log-Likelihood
        """

        X_test, Theta_test = self._check_dims(X_test, Theta_test)
        X_test = (X_test - self._X_mean) / self._X_std
        S, n, p = X_test.shape

        Omega_mat = self.Omega_mat
        Omega_diag = torch.diag(Omega_mat)
        XM_mat = X_test - self.mu_vec
        Z_tens = Theta_test @ torch.unsqueeze(XM_mat, dim=1)
        Rho_mat = torch.tensordot(Z_tens, self.Drho_tens, dims=([1, 3], [0, 2]))
        A_mat = (Rho_mat + XM_mat @ Omega_mat) / Omega_diag
        pseudo_logLik = -0.5 * (
                S * n * p * np.log(2 * np.pi)
                - S * n * torch.sum(torch.log(Omega_diag))
                + torch.sum(A_mat ** 2 * Omega_diag)
        )
        return pseudo_logLik

    def information_criterion(self, which: Union[str, List[str]]) -> List[torch.Tensor]:
        r"""
        Calculates AIC and/or BIC given the currently estimated parameters and training
        data.

        Based on the pseudo-log-likelihood of the training data set and the number of
        estimated parameters not equal to zero, Akaike Information Criterion (AIC) and
        Bayesian Information Criterion (BIC) can be calculated.

        Parameters
        ----------
        which : str or list
            Which information criterion should be calculated ("AIC" and/or "BIC"). If
            both are supplied in a list, they are returned in the same order.

        Returns
        -------
        list
            A list of tensors with the pseudo-log-likelihood of the training data and
            then the calculated information criterion.
        """

        if self._params is None:
            raise ValueError("Model parameters have not been estimated, yet.")
        if isinstance(which, str):
            which = [which]

        # temporarily revert standardization -> will be applied in score again
        pseudo_logLik = self.score(self.X_train * self._X_std + self._X_mean, self.Theta_train)
        results = [pseudo_logLik]
        for criterion in which:
            n_params = torch.sum(self._params != 0)
            if criterion == "AIC":
                criterion_value = -2 * pseudo_logLik + 2 * n_params
            elif criterion == "BIC":
                criterion_value = -2 * pseudo_logLik + n_params * np.log(self.S * self.n * self.p)
            else:
                raise ValueError("invalid information criterion")
            results.append(criterion_value)
        return results

    def predict(
            self,
            X_mat: torch.Tensor,
            Theta_tens: torch.Tensor,
            density: str = "observation"
    ) -> torch.Tensor:
        r"""
        Predict the data of each observation in `X_mat` given all other observations
        and `Theta_tens`. Predictions are based on the current parameter estimates.

        Data tensors should have data type `torch.float64`. If only one sample was
        observed ($S = 1$) or the expansion has order 1 ($L = 1$) some information
        may be inferred from low-dimensional tensors, but it is safer to provide the
        tensors with respective singleton dimensions as described below.

        Parameters
        ----------
        X_test : torch.Tensor
            A $S \times n \times p$ tensor with the observed variable data.
        Theta_test : torch.Tensor
            A $S \times L \times n \times n$ tensor with the spatial information.
        density : str
            Which density the predictions are based on. Options are `observation`
            and `variable. Defaults to `observation`.

        Returns
        -------
        torch.Tensor
            A tensor of shape $S \times n \times p$ containing the expected values.
        """
        X_mat, Theta_tens = self._check_dims(X_mat, Theta_tens)
        X_mat = (X_mat - self._X_mean) / self._X_std

        if density == "observation":
            XM_mat = X_mat - self.mu_vec
            Z_tens = Theta_tens @ torch.unsqueeze(XM_mat, dim=1)
            Rho_mat = torch.tensordot(Z_tens, self.Drho_tens, dims=([1, 3], [0, 2]))
            X_hat = -1 * Rho_mat @ torch.inverse(self.Omega_mat) + self.mu_vec
        elif density == "variable":
            XM_mat = X_mat - self.mu_vec
            Z_tens = Theta_tens @ torch.unsqueeze(XM_mat, dim=1)
            Rho_mat = torch.tensordot(Z_tens, self.Drho_tens, dims=([1, 3], [0, 2]))
            A_mat = (Rho_mat + XM_mat @ self.Omega_mat) / torch.diag(self.Omega_mat)
            X_hat = XM_mat - A_mat + self.mu_vec
        else:
            raise ValueError("Unknown `density`. Choose from `['full', 'observation', 'variable']`!")
        X_hat = X_hat * self._X_std + self._X_mean
        return X_hat


class SpaCeNetGridSearch:
    r"""Perform a grid search cross-validation for SpaCeNet.

    SpaCeNet fits a sparse inverse covariance matrix for spatially distributed data. It
    not only models associations of variables within observations but also between
    different observations.


    Attributes
    ----------
    errors : list
        A list of errors that may have been encountered during the grid search process.
    results : pandas.DataFrame
        A data frame with the results of the grid search.


    Methods
    -------
    fit(X_train, Theta_train, X_test = None, Theta_test = None)
        Performs a grid search for the observed training and test sets $X$ and $\Theta$.
    """

    def __init__(
            self,
            min_alpha: float = 1e-5,
            max_alpha: float = 10.,
            n_alpha: int = 4,
            min_beta: float = 1e-5,
            max_beta: float = 10.,
            n_beta: int = 4,
            n_refinements: int = 6,
            refinement_criterion: str = "AIC",
            penalize_diag: bool = False,
            warm_start: bool = False,
            verbose: int = 1,
            device: torch.device = "cpu",
            optimizer_options: Optional[dict] = None
    ) -> None:
        r"""
        Parameters
        ----------
        min_alpha : float
            The minimal $\alpha$ value (> 0) that will be evaluated.
        max_alpha : float
            The maximal $\alpha$ value that will be evaluated.
        n_alpha : int
            The number of logarithmically distributed $\alpha$ values between `min_alpha`
            and `max_alpha` that will be evaluated.
        min_beta : float
            The minimal $\beta$ value (> 0) that will be evaluated.
        max_beta : float
            The maximal $\beta$ value that will be evaluated.
        n_beta : int
            The number of logarithmically distributed $\beta$ values between `min_beta`
            and `max_beta` that will be evaluated.
        n_refinements : int
            The number of grid refinements that will be performed.
        refinement_criterion : str
            The criterion based on which the currently best hyperparameters are selected
            to refine the grid. Options are "score_test" (test data is needed!), "AIC" and
            "BIC".
        penalize_diag : bool
            Whether or not the diagonal of $\Omega$ (and therefore $\Lambda$) should be
            penalized as well.
        warm_start : bool
            Whether or not the last estimated parameters should be used as initial
            parameters for a new set of hyperparameters. This might speed up the grid
            search significantly.
        verbose : int
            Different verbosity levels are available:
            - 0: silent
            - 1: display a `tqdm` progress bar
            - 2: display a `tqdm` progress bar and print status messages about the current
                 set of hyper parameters.
        device : torch.device
            The device where all calculations are performed. Must match the device of
            training and test data.
        optimizer_options : dict, optional
            A dictionary with keywords and arguments that are passed on to the optimizer.
        """

        if optimizer_options is None:
            self.optimizer_options = dict()
        else:
            self.optimizer_options = optimizer_options
        self.optimizer_options["verbose"] = False
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.n_alpha = n_alpha
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.n_beta = n_beta
        self.n_refinements = n_refinements
        self.refinement_criterion = refinement_criterion
        self.penalize_diag = penalize_diag
        self.warm_start = warm_start
        self.verbose = verbose
        self.device = device
        self.errors = []
        self.results = None

    def __repr__(self):
        return f"""SpaCeNetGridSearch(
      min_alpha={self.min_alpha!r},
      max_alpha={self.max_alpha!r},
      n_alpha={self.n_alpha!r},
      min_beta={self.min_beta!r},
      max_beta={self.max_beta!r},
      n_beta={self.n_beta!r},
      n_refinements={self.n_refinements!r},
      refinement_criterion={self.refinement_criterion!r},
      include_zero={self.include_zero!r},
      penalize_diag={self.penalize_diag!r},
      warm_start={self.warm_start!r},
      verbose={self.verbose!r},
      device={self.device!r},
      optimizer_options={self.optimizer_options!r}
    )"""

    def fit(
            self,
            X_train: torch.Tensor,
            Theta_train: torch.Tensor,
            X_test: Optional[torch.Tensor] = None,
            Theta_test: Optional[torch.Tensor] = None
    ) -> pd.DataFrame:
        r"""
        Performs a grid search for the observed training and test sets $X$ and $\Theta$.
        If the test set is ommitted, only AIC and BIC are calculated and the score (i.e.
        the pseudo-log-likelihood) for the test data will be filled with nan.
        Any errors encountered during the grid search are written to `self.errors`.
        Further details may be gathered from the `SpaCeNet` class.


        Parameters
        ----------
        X_train : torch.Tensor
            A $S \times n \times p$ tensor with the observed variable data to train the
            models.
        Theta_train : torch.Tensor
            A $S \times L \times n \times n$ tensor with the spatial information to train
            the models.
        X_test : optional, torch.Tensor
            A tensor with the observed variable data to calculate the test score of the
            models.
        Theta_test : optional, torch.Tensor
            A tensor with the spatial information to calculate the test score of the
            models.


        Returns
        -------
        pandas.DataFrame
            Results are returned as a data frame (also written to `self.results`).
        """

        model = SpaCeNet(
            alpha=None,
            beta=None,
            device=self.device,
            penalize_diag=self.penalize_diag,
            optimizer_options=self.optimizer_options
        )

        result_df = pd.DataFrame({
            "S_train": pd.Series(dtype=int),
            "n_train": pd.Series(dtype=int),
            "p": pd.Series(dtype=int),
            "L": pd.Series(dtype=int),
            "alpha": pd.Series(dtype=float),
            "beta": pd.Series(dtype=float),
            "max_iter": pd.Series(dtype=int),
            "n_iter": pd.Series(dtype=int),
            "step_size": pd.Series(dtype=float),
            "momentum": pd.Series(dtype=float),
            "eps_target": pd.Series(dtype=float),
            "eps_final": pd.Series(dtype=float),
            "score_test": pd.Series(dtype=float),
            "score_train": pd.Series(dtype=float),
            "AIC": pd.Series(dtype=float),
            "BIC": pd.Series(dtype=float),
            "num_params_mu": pd.Series(dtype=int),
            "num_params_Omega": pd.Series(dtype=int),
            "num_params_Drho": pd.Series(dtype=int),
        })
        min_alpha, max_alpha = self.min_alpha, self.max_alpha
        min_beta, max_beta = self.min_beta, self.max_beta
        refinement_iterator = range(self.n_refinements)
        if self.verbose > 0:
            refinement_iterator = tqdm(refinement_iterator, desc="refinement")
        for refinement in refinement_iterator:
            alphas = np.logspace(np.log10(min_alpha), np.log10(max_alpha), self.n_alpha)
            betas = np.logspace(np.log10(min_beta), np.log10(max_beta), self.n_beta)
            grid = np.array([[a, b] for a in alphas for b in betas])
            refinement_results = []

            if self.verbose > 0:
                grid = tqdm(grid, desc="grid point", leave=False)
            for alpha, beta in grid:
                if self.verbose > 1:
                    tqdm.write(f"fit {alpha = :.2e}, {beta = :.2e}")

                model.alpha = alpha
                model.beta = beta
                try:
                    model.fit(
                        X_train,
                        Theta_train,
                        reset_params=not self.warm_start
                    )
                    score_train, AIC, BIC = model.information_criterion(["AIC", "BIC"])
                    num_params_mu = torch.sum(model._params[:model.p] != 0.).item()
                    num_params_Omega = torch.sum(model._params[model.p:(model.p + model.n_triu)] != 0.).item()
                    num_params_Drho = torch.sum(model._params[(model.p + model.n_triu):] != 0.).item()

                    if (X_test is not None) and (Theta_test is not None):
                        score_test = model.score(X_test, Theta_test).cpu().item()
                    else:
                        score_test = float("nan")

                    iter_result = {
                        "S_train": model.S,
                        "n_train": model.n,
                        "p": model.p,
                        "L": model.L,
                        "alpha": model.alpha,
                        "beta": model.beta,
                        "max_iter": model.optimizer_options.get("max_iter"),
                        "n_iter": model.n_iter,
                        "step_size": model.optimizer_options.get("step_size"),
                        "momentum": model.optimizer_options.get("momentum"),
                        "eps_target": model.optimizer_options.get("eps"),
                        "eps_final": model.final_eps.item(),
                        "score_test": score_test,
                        "score_train": score_train.item(),
                        "AIC": AIC.item(),
                        "BIC": BIC.item(),
                        "num_params_mu": num_params_mu,
                        "num_params_Omega": num_params_Omega,
                        "num_params_Drho": num_params_Drho,
                    }
                    iter_error = ""

                except Exception:
                    iter_error = f"failed for {alpha = }, {beta = } with the following traceback:\n"
                    iter_error += tb.format_exc()
                    iter_result = {
                        "S_train": model.S,
                        "n_train": model.n,
                        "p": model.p,
                        "L": model.L,
                        "alpha": model.alpha,
                        "beta": model.beta,
                        "max_iter": model.optimizer_options.get("max_iter"),
                        "eps_target": model.optimizer_options.get("eps"),
                        "step_size": model.optimizer_options.get("step_size"),
                        "momentum": model.optimizer_options.get("momentum")
                    }

                refinement_results.append(iter_result)
                if iter_error:
                    self.errors.append(iter_error)

            result_df = pd.concat([result_df, pd.DataFrame(refinement_results)], ignore_index=True)
            result_df.drop_duplicates(subset=["alpha", "beta"], inplace=True)

            # find new hyperparameter bounds for next refinement step
            if (self.refinement_criterion == "score_test"):
                if (X_test is None) or (Theta_test is None):
                    raise ValueError("Test data must be provided if refinement_criterion == 'score_test'.")
                best_alpha, best_beta = result_df[["alpha", "beta"]].iloc[np.nanargmax(result_df["score_test"])]
            elif self.refinement_criterion in ["AIC", "BIC"]:
                best_alpha, best_beta = result_df[["alpha", "beta"]].iloc[
                    np.nanargmin(result_df[self.refinement_criterion])]
            else:
                raise ValueError("Unknown 'refinement_criterion'!")

            all_alphas = sorted(result_df.alpha.unique())
            all_betas = sorted(result_df.beta.unique())
            best_alpha_ind = np.argwhere(np.array(all_alphas) == best_alpha)[0, 0]
            best_beta_ind = np.argwhere(np.array(all_betas) == best_beta)[0, 0]

            best_alpha_on_lower_bound = (best_alpha_ind == 0)
            best_alpha_on_upper_bound = (best_alpha_ind == len(all_alphas) - 1)
            best_beta_on_lower_bound = (best_beta_ind == 0)
            best_beta_on_upper_bound = (best_beta_ind == len(all_betas) - 1)

            if best_alpha_on_lower_bound:
                min_alpha = best_alpha
                max_alpha = np.logspace(np.log10(all_alphas[0]), np.log10(all_alphas[1]), 3)[1]
            elif best_alpha_on_upper_bound:
                min_alpha = np.logspace(np.log10(all_alphas[-2]), np.log10(all_alphas[-1]), 3)[1]
                max_alpha = best_alpha
            else:
                min_alpha = \
                np.logspace(np.log10(all_alphas[best_alpha_ind - 1]), np.log10(all_alphas[best_alpha_ind]), 3)[1]
                max_alpha = \
                np.logspace(np.log10(all_alphas[best_alpha_ind]), np.log10(all_alphas[best_alpha_ind + 1]), 3)[1]
            if best_beta_on_lower_bound:
                min_beta = best_beta
                max_beta = np.logspace(np.log10(all_betas[0]), np.log10(all_betas[1]), 3)[1]
            elif best_beta_on_upper_bound:
                min_beta = np.logspace(np.log10(all_betas[-2]), np.log10(all_betas[-1]), 3)[1]
                max_beta = best_beta
            else:
                min_beta = np.logspace(np.log10(all_betas[best_beta_ind - 1]), np.log10(all_betas[best_beta_ind]), 3)[1]
                max_beta = np.logspace(np.log10(all_betas[best_beta_ind]), np.log10(all_betas[best_beta_ind + 1]), 3)[1]

        self.results = pd.DataFrame(result_df)
        return self.results
