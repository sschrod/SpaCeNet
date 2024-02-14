from math import isnan
from typing import Tuple
import torch


def solve(
        loss_grad: callable,
        prox: callable,
        params_init: torch.Tensor,
        step_size: float = 1E-6,
        momentum: float = 0.9,
        eps: float = 1e-5,
        verbose: bool = True,
        max_iter: int = 10_000,
        max_step_size_reductions: int = 3,
        step_size_reduction_factor: float = 10.
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implements an gradient descent algorithm including Nesterov acceleration (`momentum`)
    and proximal operators.

    Gradients are calculated with Nesterov acceleration. After each gradient step, the
    parameters are updated according to the `prox` function.
    If the step_size is chosen too large, the parameters may cause overflows (divergence
    instead of convergence). The algorithm can be automatically started again with a lower
    `step_size` (see `max_step_size_reductions` and `step_size_reduction_factor`).

    Parameters
    ----------
    loss_grad : callable
        A function that returns a loss and gradient for a given parameter vector in that
        order.
    prox : callable
        A function that applies the proximal operator to the parameter vector and returns
        the updated parameters.
    params_init : torch.Tensor
        The initial parameters.
    step_size : float
        The step size or learn rate. Determines how much the parameters change each
        iteration. Defaults to `1e-6`.
    momentum : float
        The momentum parameter for nesterov acceleration. Defaults to `0.9`.
    eps : float
        Threshold to exit the algorithm. If the relative change in loss between two
        iterations is lower than `eps` the algorithm is terminated. Defaults to `1e-5`.
    verbose : boolean
        Whether or not fitting updates should be printed. Defaults to `True`.
    max_iter : int
        The maximum number of iterations after which the algorithm will be terminated.
        Defaults to 10,000.
    max_step_size_reductions : int
        If overflows are encountered, the fitting will be restarted for up to
        `max_step_size_reductions` times with a lower `step_size`. Defaults to 3.
    step_size_reduction_factor : float
        If overflows are encountered and the fit is restarted, the `step_size` is divided
        by this number. Defaults to 10.

    Returns
    -------
    tuple
        A tuple of tensors with the optimized parameter vector and the losses per
        iteration.
    """

    params = torch.clone(params_init)
    velocity = torch.zeros_like(params)
    losses = torch.zeros(max_iter + 1, device=params_init.device, dtype=params_init.dtype)
    losses[0] = float("inf")
    for i in range(1, max_iter + 1):
        projection = params + momentum * velocity
        losses[i], gradient = loss_grad(projection)
        relative_loss_change = abs((losses[i] - losses[i - 1]) / losses[i - 1])
        if isnan(relative_loss_change ** (i - 1)):  # skip first iter, where nan is guaranteed
            if max_step_size_reductions == 0:
                raise RuntimeError("relative_loss_change is nan. Possibly due to overflow.")
            if verbose:
                print(f"Restart with lower step_size = {step_size / step_size_reduction_factor:.2e}")
            return solve(
                loss_grad=loss_grad,
                prox=prox,
                params_init=params_init,
                step_size=step_size / step_size_reduction_factor,
                momentum=momentum,
                eps=eps,
                verbose=verbose,
                max_iter=max_iter,
                max_step_size_reductions=max_step_size_reductions - 1,
                step_size_reduction_factor=step_size_reduction_factor
            )
        if verbose and i % 100 == 0:
            print(f"iter {i}, projection loss = {losses[i]}, {relative_loss_change = :.2e}")
        if relative_loss_change < eps:
            break
        velocity = momentum * velocity - step_size * gradient
        params += velocity
        params = prox(params, step_size)
    if verbose:
        print(f"exit after iter {i}, projection_loss = {losses[i]}, {relative_loss_change = :.2e}")
    return params, losses[1:(i + 1)]
