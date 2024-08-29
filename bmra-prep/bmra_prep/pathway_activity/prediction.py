# Future imports
from __future__ import annotations

from typing import Tuple

# Third party imports
import numpy as np
import numpy.typing as npt
import torch


def calc_pathway_activity(
    X: npt.NDArray[np.float64], coeffs: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Calculate pathway activities based on coefficients.

    Pathway activities are calculated as a simple linear combination, with a fixed
    offset of 1.

    Parameters
    ----------
    X : np.ndarray
      Data.
    coeffs : np.ndarray
      Coefficients for calculating pathway activity.

    Returns
    -------
    pathway_activity : np.ndarray
    """
    return (coeffs @ X) + 1


def predict_coeffs(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    mask: npt.NDArray[np.float64],
    n_iterations: int,
    weight_lm: float,
    weight_orth: float,
    weight_lasso: float,
    weight_neg_pen: float,
    lr: float = 0.0001,
) -> npt.NDArray[np.float64]:
    r"""
    Calculate coefficients for pathway activity based on custom loss function.

    Uses stochastic gradient descend from pytorch to find minimum of loss function. The
    loss function is a combination of multiple linear regressions, a Lasso
    regularization, a orthogonality criteria and a restriction to predict only positive
    values on the data set.

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    mask : np.ndarray
    n_iterations : int
    weight_lm : float
    weight_orth : float
    weight_lasso : float
    weight_neg_pen : float
    lr : float
      Learning rate for gradient descent.

    Returns
    -------
    coeffs : np.ndarray
      Best set of coefficients after minimization.

    Notes
    -----
    The following loss function is optimized for to find the coefficients A:

    .. math::

        L (X, A) = w_{LM}  \sum mask * (Y-(AX + 1))^2 + w_{orth} \sum_{i1, i2} \frac{(\sum_k a_{i1k} \dot a_{i2k})^2}{|a_{i1}|^2_2 |a_{i2}|^2_2} + w_{lasso} \sum |A| + w_{pen} \sum |min(AX + 1, 0)|

    """
    X = torch.asarray(X, dtype=torch.float64)
    y = torch.asarray(y, dtype=torch.float64)
    mask = torch.asarray(mask, dtype=torch.float64)

    a_shape = (y.shape[0], X.shape[0])

    m = Model(a_shape)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    loss_fun = CustomLoss(mask, weight_lm, weight_orth, weight_lasso, weight_neg_pen)
    losses, coeffs = training_loop(
        X=X,
        y_true=y,
        model=m,
        loss_fun=loss_fun,
        optimizer=opt,
        n_iterations=n_iterations,
    )

    return coeffs


class Model(torch.nn.Module):
    """
    Custom pytorch model: Linear model
    """

    def __init__(self, coeff_shape: Tuple[int, int]):
        super().__init__()

        coeffs = (
            torch.distributions.Uniform(-0.1, 0.1)
            .sample(torch.Size(coeff_shape))
            .to(torch.float64)
        )
        self.coeffs = torch.nn.Parameter(coeffs)

    def forward(self, X: torch.Tensor):
        # fixed intercept = 1
        return (self.coeffs @ X) + 1


class CustomLoss(torch.nn.Module):
    """
    Custom loss function.
    """

    def __init__(
        self,
        perturbation_mask: torch.Tensor,
        weight_lm: float = 1.0,
        weight_orth: float = 1.0,
        weight_lasso: float = 1.0,
        weight_neg_pen: float = 1.0,
    ):
        super().__init__()

        self.perturbation_mask = perturbation_mask
        # set weights for terms of cost function
        self.weight_lm, self.weight_orth, self.weight_lasso, self.weight_neg_pen = (
            weight_lm,
            weight_orth,
            weight_lasso,
            weight_neg_pen,
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, coeffs: torch.Tensor):
        # linear regression: sum of squares
        lin_reg = (
            self.weight_lm
            * (self.perturbation_mask * torch.square(y_true - y_pred)).sum()
        )

        # orthogonality of coeffs
        dotprod_matrix = coeffs @ coeffs.T
        norms = torch.square(torch.linalg.norm(coeffs, axis=1))
        norm_matrix = torch.outer(norms, norms)
        # unpacking necessary to enforce number of elements to mypy
        _row, _col = norm_matrix.shape
        triu_indx = torch.triu_indices(row=_row, col=_col, offset=1)
        row_indx = triu_indx[0]
        col_indx = triu_indx[1]

        orth = self.weight_orth * torch.sum(
            torch.square(dotprod_matrix[row_indx, col_indx])
            / norm_matrix[row_indx, col_indx]
        )

        # lasso
        lasso = self.weight_lasso * torch.abs(coeffs).sum()

        # penalty for negatively predicted values
        pen = self.weight_neg_pen * torch.where(y_pred < 0, -y_pred, 0).sum()

        return lin_reg + orth + pen + lasso


def training_loop(
    X: torch.Tensor,
    y_true: torch.Tensor,
    model: Model,
    loss_fun: CustomLoss,
    optimizer: torch.optim.Optimizer,
    n_iterations: int = 100_000,
):
    "Training loop for torch model."
    losses = np.full(n_iterations, np.inf)
    best_weights = torch.empty(model.coeffs.shape, dtype=torch.float64)
    for i in range(n_iterations):
        preds = model(X)
        loss = loss_fun(preds, y_true, model.coeffs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # update weights if better
        loss_np = loss.detach().numpy()
        if loss_np < losses.min():
            best_weights = model.coeffs.clone()
        losses[i] = loss_np

    return losses, best_weights.detach().numpy()
