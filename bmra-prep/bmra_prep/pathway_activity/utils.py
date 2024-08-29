# Future imports
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

# Local imports
from ..input_classes import GlobalResponse


def calc_global_response_from_pathway_activity(
    pathway_activity: npt.NDArray[np.float64],
    modules: List[str],
    perturbations: List[str] | None = None,
) -> GlobalResponse:
    r"""
    Calculate global responses based on predicted pathway activities.

    Returns a `GlobalResponse` object.

    Parameters
    ----------
    pathway_activity : np.ndarray
    modules : List[str]
    perturbations : List[str]

    Returns
    -------
    GlobalResponse

    Notes
    -----
    The following formula is used to calculate global responses based on predicted
    pathway activities :math:`f`:

    .. math::

        R = 2 \frac{f - 1}{f + 1}
    """
    global_response = 2 * (pathway_activity - 1) / (pathway_activity + 1)
    return GlobalResponse.from_numpy(global_response, modules, perturbations)


def process_inhibitor_data(
    inhib_conc_df: pd.DataFrame, ic50_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r"""
    Calculate perturbation matrix and expected activity values.

    Parameters
    ----------
    inhib_conc_df : pd.DataFrame
      Contains information about the used drug concentrations and in which experiment
      they were used. Required columns: `pert_name`, `drug`, `dose`
    ic50_df : pd.DataFrame
      Contains information about the used drugs, specifically the IC50 values per
      module. For all combinations of drug and module which are not present in the
      dataframe, it is assumed that the drug does have no effect on the module. Required
      columns: `drug`, `module`, `ic50`

    Returns
    -------
    y_true : pd.DataFrame
      The expected activity values, based on drug concentrations and IC50 values.
    pert_df : pd.DataFrame
      The perturbation matrix in dataframe format.

    Notes
    -----
    The following formula is used to calculate `y_true`:

    .. math::

        y_{true} = \frac{1}{1 + \frac{conc}{ic50}}
    """
    # check that inputs have the correct column names
    for colname in ["pert_name", "drug", "dose"]:
        if colname not in inhib_conc_df.columns:
            raise ValueError(f"Missing column in inhib_conc_df: {colname}")
    for colname in ["module", "drug", "ic50"]:
        if colname not in ic50_df.columns:
            raise ValueError(f"Missing column in ic50_df: {colname}")

    modules = ic50_df["module"].unique().tolist()
    perts = inhib_conc_df["pert_name"].unique().tolist()
    n_modules = len(modules)
    n_perts = len(perts)

    # parse dataframes to modules x perturbations matrices
    inhib_conc_matrix = np.zeros((n_modules, n_perts))
    ic50_matrix = np.ones((n_modules, n_perts))

    for i, module in enumerate(modules):
        drugs_per_module = ic50_df["drug"][ic50_df["module"] == module].tolist()
        for drug in drugs_per_module:
            ic50 = ic50_df["ic50"][
                (ic50_df["drug"] == drug) & (ic50_df["module"] == module)
            ].values
            if ic50.size != 1:
                raise ValueError(
                    f"Multiple entries of IC5O for drug {drug} in module {module}"
                )
            perts_with_drug = inhib_conc_df["pert_name"][
                inhib_conc_df["drug"] == drug
            ].tolist()

            for pert in perts_with_drug:
                j = perts.index(pert)
                inhib_conc = inhib_conc_df["dose"][
                    (inhib_conc_df["pert_name"] == pert)
                    & (inhib_conc_df["drug"] == drug)
                ].values
                if inhib_conc.size != 1:
                    raise ValueError(
                        f"Multiple entries of inhibitor concentration for drug {drug} "
                        "in perturbation {pert}"
                    )
                inhib_conc_matrix[i, j] = inhib_conc.item()
                ic50_matrix[i, j] = ic50.item()

    # process matrices to outputs
    y_true_df = pd.DataFrame(
        1 / (1 + inhib_conc_matrix / ic50_matrix),
        index=modules,
        columns=perts,
    )
    pert_df = pd.DataFrame(
        np.where(inhib_conc_matrix != 0, 1, 0),
        index=modules,
        columns=perts,
    )

    return y_true_df, pert_df
