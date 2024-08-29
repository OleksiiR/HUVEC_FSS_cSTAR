"""
Submodule for managing BMRA inputs in separate classes.

The available classes are `GlobalResponse`, `PerturbationMatrix` and `Network`.
"""


from __future__ import annotations

import abc

from typing import List, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd

Self = TypeVar("Self", bound="DataModulesXPerturbations")


class DataModulesXPerturbations(abc.ABC):
    def __init__(self, _data: pd.DataFrame):
        if len(set(_data.columns)) != len(_data.columns):
            raise ValueError("No duplicates in perturbation names.")
        if len(set(_data.index)) != len(_data.index):
            raise ValueError("No duplicates in module names.")
        self._data = _data

    @classmethod
    def from_numpy(
        cls,
        data: npt.NDArray,
        modules: List[str] | None = None,
        perturbations: List[str] | None = None,
    ):
        """Construct instance from numpy array."""
        return cls.from_pandas(pd.DataFrame(data, index=modules, columns=perturbations))

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        modules: List[str] | None = None,
        perturbations: List[str] | None = None,
    ):
        """Construct instance from pandas df."""
        df = df.copy()
        if modules is not None:
            df.index = pd.Index(modules)
        if perturbations is not None:
            df.columns = pd.Index(perturbations)

        return cls(df)

    @property
    def modules(self) -> List:
        """Get module annotation of global response."""
        return self._data.index.tolist()

    @property
    def perturbations(self) -> List:
        """Get perturbation annotation of global response."""
        return self._data.columns.tolist()

    @property
    def values(self) -> npt.NDArray:
        """Get underlying numerical values."""
        return self._data.values

    @property
    def dataframe(self) -> pd.DataFrame:
        """Get underlying pandas data frame."""
        return self._data

    @abc.abstractmethod
    def combine(self, other: Self) -> Self:
        ...

    def _combine(self, other: Self) -> Self:
        """
        Combine two objects of this abstract base class.

        Columns(perturbations) in `other` will be filtered by what is present in `self`.

        Parameters
        ----------
        other : Self

        Returns
        -------
        Self
        """
        if type(self) is not type(other):
            raise ValueError(
                f"Can not combine object of type {type(self)} with object of type "
                f"{type(self)}"
            )

        data_self = self._data
        data_other = other._data.T  # transposed for easier filtering and sorting

        perts_self = self.perturbations

        data_other = data_other[data_other.index.isin(perts_self)]

        # sort by perts list
        data_other["sort_col"] = data_other.index.map(
            {val: i for i, val in enumerate(perts_self)}
        )

        data_other = data_other.sort_values("sort_col")
        data_other = data_other.drop("sort_col", axis=1)

        # transpose back
        data_other = data_other.T

        # quality check inputs
        perts_other = data_other.columns.tolist()
        if set(perts_self) != set(perts_other):
            msg = (
                f"Not all perturbation experiments in self present in other: \n"
                f"{set(perts_self) - set(perts_other)}"
            )
            raise ValueError(msg)

        return self.__class__.from_pandas(pd.concat([data_self, data_other]))


class PerturbationMatrix(DataModulesXPerturbations):
    def __init__(self, _data: pd.DataFrame):
        _data = _data.astype(np.int8)
        if not np.all((_data.values == 0) | (_data.values == 1)):
            raise ValueError("Perturbation matrix must be binary (only 0 and 1).")
        super().__init__(_data)

    @classmethod
    def new_empty(
        cls, modules: List[str], perturbations: List[str]
    ) -> PerturbationMatrix:
        """Construct an instance with only zero values."""
        return cls.from_numpy(
            data=np.zeros((len(modules), len(perturbations))),
            modules=modules,
            perturbations=perturbations,
        )

    def combine(self, other: Self) -> Self:
        """
        Combine two perturbation matrices.

        Columns(perturbations) in `other` will be filtered by what is present in `self`.

        Parameters
        ----------
        other : PerturbationMatrix

        Returns
        -------
        PerturbationMatrix
        """
        return self._combine(other)


class GlobalResponse(DataModulesXPerturbations):
    """
    Global responses class.

    Holds data for global responses and offers useful methods for annotation and
    manipulation.
    """

    def __init__(self, _data):
        _data = _data.astype(np.float64)
        super().__init__(_data)

    def combine(self, other: Self) -> Self:
        """
        Combine two global response matrices.

        Columns(perturbations) in `other` will be filtered by what is present in `self`.

        Parameters
        ----------
        other : GlobalResponse

        Returns
        -------
        GlobalResponse
        """
        return self._combine(other)


class Network:
    def __init__(self, _adj_df: pd.DataFrame):
        _adj_df = _adj_df.astype(np.int8)

        if not np.all(_adj_df.index == _adj_df.columns):
            raise ValueError("Expected identical row and column names.")
        if not np.all((_adj_df.values == 0) | (_adj_df.values == 1)):
            raise ValueError("Network must be binary, only 0 and 1 allowed as values.")

        self._adj_df = _adj_df

    @classmethod
    def from_adjacency_matrix(
        cls, adj_mat: npt.ArrayLike, nodes: List[str] | None = None
    ) -> Network:
        """
        Create Network instance from adjacency matrix.

        Parameters
        ----------
        adj_mat : npt.ArrayLike
          Adjacency matrix.
        nodes : List[str]
          List of node names.

        Returns
        -------
        Network
        """
        adj_mat = np.asarray(adj_mat, dtype=np.int8)

        if adj_mat.shape[0] != adj_mat.shape[1] or adj_mat.ndim != 2:
            raise ValueError("Adjacency matrix should be 2D square matrix.")

        return Network(pd.DataFrame(adj_mat, columns=nodes, index=nodes))

    @classmethod
    def from_pandas(cls, df: pd.DataFrame, nodes: List[str] | None = None) -> Network:
        """
        Create Network instance from adjacency matrix in pandas df.
        """
        if nodes is not None:
            df.index = pd.Index(nodes)
            df.columns = pd.Index(nodes)
        return cls(df)

    @classmethod
    def from_edge_list(
        cls, edge_list: pd.DataFrame, nodes: List[str] | None
    ) -> Network:
        """
        Create Network instance from edge list.

        Parameters
        ----------
        edge_list : pd.DataFrame
          Data frame, from which the first two columns will be interpreted as "from" and
          "to". Other columns will be ignored.
        nodes : List[str] | None
          List of nodes in the network. If given, network will be arrange in the order
          of the given nodes, and nodes which do not appear in the edge list will be
          inserted.

        Returns
        -------
        Network
        """
        # TODO
        raise NotImplementedError

    @property
    def modules(self) -> List[str]:
        """Get modules of Network instance."""
        return self._adj_df.index.tolist()

    @property
    def adj_mat(self) -> npt.NDArray[np.int8]:
        """Get adjacency matrix as numpy array."""
        return self._adj_df.copy().values

    @property
    def transposed_adj_mat(self) -> npt.NDArray[np.int8]:
        """Get transposed adjacency matrix as numpy array."""
        return self._adj_df.copy().values.T

    @property
    def annotated_adj_mat(self) -> pd.DataFrame:
        """Get annotated adjacency matrix as pandas data frame."""
        return self._adj_df.copy()

    @property
    def annotated_transposed_adj_mat(self) -> pd.DataFrame:
        """Get annotated transposed adjacency matrix as pandas data frame."""
        return self._adj_df.copy().T

    def reorder_modules(self, module_list: List):
        """Reorder modules by list."""
        if set(module_list) != set(self.modules):
            raise ValueError()  # TODO
        raise NotImplementedError  # TODO


def bmra_inputs(
    global_response: GlobalResponse,
    perturbation_matrix: PerturbationMatrix,
    prior_network: Network,
    forbidden_network: Network | None = None,
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.int8],
    npt.NDArray[np.int8],
    npt.NDArray[np.int8],
]:
    """
    Transform `bmra_prep` objects into the matrices expected by `run_bmra`.

    Fails if inputs are not consistently annotated.

    Parameters
    ----------
    global_reponse : GlobalResponse
      Annotated global response object.
    perturbation_matrix : PerturbationMatrix
      Annotated perturbation matrix object.
    prior_network : Network
      Annotated prior network.
    forbidden_network : Network | None
      Annotated forbidden network. If None, will allow all connections that are not
      from a node to itself.

    Returns
    -------
    data : np.ndarray
      Global response matrix.
    pert : np.ndarray
      Perturbation matrix.
    G : np.ndarray
      Transposed network adjacency matrix of prior network.
    G_not : np.ndarray
      Transposed network adjacency matrix of prohibited interactions.
    """
    if forbidden_network is None:
        forbidden_network = Network.from_adjacency_matrix(
            adj_mat=np.eye(len(prior_network.modules)),
            nodes=prior_network.modules,
        )

    # check for consistency
    if global_response.modules != perturbation_matrix.modules:
        msg = (
            "Inconsistent module names between global responses and perturbation "
            "matrix."
        )
        raise ValueError(msg)

    if global_response.perturbations != perturbation_matrix.perturbations:
        msg = (
            "Inconsistent perturbation names between global responses and "
            "perturbation matrix"
        )
        raise ValueError(msg)

    if prior_network.modules != forbidden_network.modules:
        msg = "Inconsistent module names between prior network and forbidden network."
        raise ValueError(msg)

    if global_response.modules != prior_network.modules:
        msg = "Inconsistent module names between networks and global responses."
        raise ValueError(msg)

    # output
    return (
        global_response.values,
        perturbation_matrix.values,
        prior_network.transposed_adj_mat,
        forbidden_network.transposed_adj_mat,
    )
