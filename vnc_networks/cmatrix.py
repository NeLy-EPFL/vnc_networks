"""
Initialises the cmatrix class, standing for connectome matrix.
The purpose of this class is to manipulate the adjacency matrix of the connectome,
or subset of it, while keeping track of the associated node identities.

As much as possible, all methods are defined on the scipy sparse matrix to 
minimize the memory footprint and improve the performance of the computations.
"""

import scipy as sc
import pandas as pd

import utils.matrix_utils as matrix_utils

class CMatrix:
    def __init__(
            self,
            matrix: sc.sparse.csr_matrix,
            lookup: pd.DataFrame,
        ):
        '''
        Initialises the cmatrix class, standing for connectome matrix.

        Parameters
        ----------
        matrix : scipy.sparse.csr_matrix
            The adjacency matrix of the connectome.
        lookup : pandas.DataFrame
            The lookup table for the equivalence between node indices and node identities. 
        '''
        # verify that the matrix is a csr matrix
        if not sc.sparse.issparse(matrix):
            raise ValueError('The matrix must be a scipy sparse matrix.')
        if not isinstance(matrix, sc.sparse.csr_matrix):
            raise ValueError('The matrix must be a csr matrix.')
        self.matrix = matrix
        # verify that the lookup is a pandas dataframe
        if not isinstance(lookup, pd.DataFrame):
            raise ValueError('The lookup must be a pandas dataframe.')
        # verify that the lookup has the correct columns
        if not 'index' in lookup.columns or not 'body_id' in lookup.columns:
            raise ValueError('The lookup must have a column named "index"\
                             and a column named "body_id".')
        self.lookup = lookup
        # verify that the lookup includes indices up to the length of the matrix
        if not all(self.lookup['index'].isin(range(self.matrix.shape[0]))):
            raise ValueError('The lookup must include indices up to the length of the matrix.')
    

    # private methods
    def __restrict_subindices(self, sub_indices: list):
        '''
        Restricts the adjacency matrix and lookup to a subset of indices.

        Parameters
        ----------
        sub_indices : list
            The indices to which the adjacency matrix is restricted.
        '''
        # restrict the matrix to the indices
        self.matrix = matrix_utils.select_subset_matrix(
            self.matrix,
            sub_indices
            )
        # restrict the lookup to the indices
        self.lookup = self.lookup[self.lookup['index'].isin(sub_indices)]
        return


    # public methods
    # --- getters
    def get_matrix(self) -> sc.sparse.csr_matrix:
        '''
        Returns the adjacency matrix of the connectome.

        Returns
        -------
        scipy.sparse.csr_matrix
            The adjacency matrix of the connectome.
        '''
        return self.matrix
    
    def get_lookup(self) -> pd.DataFrame:
        '''
        Returns the lookup table for the equivalence between node indices and node identities.

        Returns
        -------
        pandas.DataFrame
            The lookup table for the equivalence between node indices and node identities.
        '''
        return self.lookup
        
    def get_body_ids(self, sub_indices: list = None) -> list:
        """
        Returns the body ids of the nodes of the adjacency matrix.

        Parameters
        ----------
        sub_indices : list, optional
            The indices of the nodes for which the body ids are returned.
            If None, the body ids of all nodes are returned. The default is None.

        Returns
        -------
        list
            The body ids of the nodes of the adjacency matrix.
        """
        if sub_indices is None:
            return self.lookup['body_id'].tolist()
        return matrix_utils.convert_index_to_bodyid(
            sub_indices,
            self.get_lookup()
            )
    
    def get_indices(self, sub_bodyid: list = None) -> list:
        """
        Returns the indices of the adjacency matrix references by the body_id list.

        Parameters
        ----------
        sub_bodyid : list, optional
            The body_ids of the nodes for which the indices are returned.
            If None, the indices of all nodes are returned. The default is None.

        Returns
        -------
        list
            The indices of the nodes of the adjacency matrix.
        """
        if sub_bodyid is None:
            return self.lookup['index'].tolist()
        return matrix_utils.convert_bodyid_to_index(
            sub_bodyid,
            self.get_lookup()
            )

    # --- setters
    def restrict_nodes(self, nodes:list):
        '''
        Restricts the adjacency matrix to a subset of nodes (body_ids).

        Parameters
        ----------
        nodes : list
            The list of nodes to which the adjacency matrix is restricted.
        '''
        # convert the nodes to indices
        indices = self.get_indices(nodes)
        # restrict the matrix amd lookup to the indices
        self.__restrict_subindices(indices)
        return
    
    def restrict_indices(self, indices:list):
        '''
        Restricts the adjacency matrix to a subset of indices.

        Parameters
        ----------
        indices : list
            The list of indices to which the adjacency matrix is restricted.
        '''
        # restrict the matrix amd lookup to the indices
        self.__restrict_subindices(indices)
        return

    # --- processing
    def power_n(self, n:int):
        '''
        Computes the n-th power of the adjacency matrix of the connectome.

        Parameters
        ----------
        n : int
            The power to which the adjacency matrix is raised.
        '''
        matrix_ = self.get_matrix()
        self.matrix = matrix_utils.connections_at_n_hops(matrix_, n)

    def within_power_n(self, n:int):
        '''
        Computes the sum of the i-th power of the adjacency matrix of the
        connectome up to n.

        Parameters
        ----------
        n : int
            The power to which the adjacency matrix is raised.
        '''
        matrix_ = self.get_matrix()
        self.matrix = matrix_utils.connections_up_to_n_hops(matrix_, n)