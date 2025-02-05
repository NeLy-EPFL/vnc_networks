#!/usr/bin/env python3
"""
Initialises the Neuron class.
Meant to be used to load data for a single neuron.
Possible use cases are visualisation, or looking at synapse distributions.

Possible upgrade: possible to match synapses on the pre and post neurons
by using the dataset "Neuprint_Synapse_Connections_manc_v1.ftr" which
has two columns: ':START_ID(Syn-ID)' and ':END_ID(Syn-ID)'.
"""

import copy
import os
import pickle
import typing
from typing import Optional

import matplotlib.axes
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from . import params
from .connectome_reader import MANC, ConnectomeReader
from .params import BodyId, NeuronAttribute
from .utils import plots_design


class Neuron:

    @typing.overload
    def __init__(
        self,
        from_file: str,
    ):...
    @typing.overload
    def __init__(
        self,
        body_id: BodyId | int,
        CR: ConnectomeReader = MANC('v1.0'),
    ):...

    def __init__(
        self,
        body_id: Optional[BodyId | int] = None,
        CR: ConnectomeReader = MANC('v1.0'),
        from_file: Optional[str] = None
    ):
        """
        Initialise the neuron.
        Loading possible either from scratch using only the body_id, or
        from a file. Loading from a file is useful if the neuron has already
        been processed and saved. Loading from scratch is useful if the neuron
        is new, but is computationally expensive and time-consuming.

        Parameters
        ----------
        CR : ConnectomeReader, optional
            The connectome reader to use.
            The default is MANC('v1.0').
        bodyId : int, optional
            The body id of the neuron.
            The default is None.
        from_file : str, optional
            The name of the file to load the neuron from.
            The default is None.
        """
        if from_file is not None:
            self.__load(from_file)
        else:
            assert (
                body_id is not None
            ), "To initialise a `Neuron`, you must provide either a `body_id` or `from_file`, but both were None."
            self.body_id = body_id
            self.CR = CR
            self.data = CR.load_data_neuron(body_id, CR.node_base_attributes())
            self.__initialise_base_attributes()

        # verify that the neuron has the required information (up to date)
        assert hasattr(self, "CR"), "CR not found in neuron."

    # private methods

    def __load(self, name: str):
        """
        Initialise the neuron from a file.
        """
        with open(os.path.join(params.NEURON_DIR, name + ".txt"), "rb") as file:
            neuron = pickle.load(file)
        self.__dict__.update(neuron)

    def __initialise_base_attributes(self):
        self.nt_type = self.data['nt_type'].values[0]
        self.nt_proba = self.data['nt_proba'].values[0]
        self.class_1 = self.data['class_1'].values[0]
        self.class_2 = self.data['class_2'].values[0]
        self.name = self.data['name'].values[0]
        self.side = self.data['side'].values[0]
        self.neuropil = self.data['neuropil'].values[0]
        self.size = self.data['size'].values[0]
        self.hemilineage = self.data['hemilineage'].values[0]

    def __load_synapse_ids(self):
        # Independently of the file structure, we should get a pd.dataframe
        # with columns ['synapse_id', 'start_bid', 'end_bid']
        
        if 'synapse_df' in self.__dict__: # already done
            return
        
        self.synapse_df = self.CR.get_synapse_df(self.body_id)

    def __load_synapse_locations(self):
        """
        Add three columns: 'X', 'Y' and 'Z' to the synapse df.

        Parameters
        ----------
        subset : list, optional
            The subset of synapse ids to convert.
            The default is None, which converts all synapse ids.
        """
        if 'synapse_df' not in self.__dict__:
            self.__load_synapse_ids()

        if "location" in self.synapse_df.columns:  # already done
            return
        
        data = self.CR.get_synapse_locations(self.synapse_df["synapse_id"].values)

        # merge with existing synapse df
        self.synapse_df = self.synapse_df.merge(
            data, on="synapse_id", how="inner"
        )
        return

    def __categorical_neuropil_information(self):
        """
        Add a 'neuropil' column to the synapse df.

        Parameters
        ----------
        subset : list, optional
            The subset of synapse ids to convert.
            The default is None, which converts all synapse ids.
        """
        if 'synapse_df' not in self.__dict__:
            self.__load_synapse_ids()
            
        if "neuropil" in self.synapse_df.columns:  # already done
            return
        
        data = self.CR.get_synapse_neuropil(self.synapse_df["synapse_id"].values)

        # merge with existing synapse df
        self.synapse_df = self.synapse_df.merge(
            data, on="synapse_id", how="inner"
        )
        return

    # public methods
    # --- getters
    def get_data(self):
        return self.data

    def get_synapse_table(self):
        """
        Get the synapse table of the connections.
        """
        if 'synapse_df' not in self.__dict__:
            self.__load_synapse_ids()
        return copy.deepcopy(self.synapse_df)

    def get_synapse_distribution(self, threshold: bool = False):
        """
        Get the synapse distribution for the neuron.

        Parameters
        ----------
        threshold : bool, optional
            Whether to apply a threshold to the synapse distribution.
            The default is False.
        """
        # check if the synapse df is loaded
        self.__load_synapse_ids()
        self.__load_synapse_locations()

        # threshold if for a given neuron, the number of synapses is below the threshold
        if threshold:
            syn_count = self.synapse_df.groupby("end_bid").size().reset_index()
            syn_count.columns = ["end_bid", "syn_count"]
            syn_count = syn_count[syn_count["syn_count"] >= params.SYNAPSE_CUTOFF]
            to_keep = syn_count["end_bid"].values
            synapses = self.synapse_df[self.synapse_df["end_bid"].isin(to_keep)]
        else:
            synapses = self.synapse_df
        # get the synapse positions
        X = synapses["X"].values
        Y = synapses["Y"].values
        Z = synapses["Z"].values
        # save the positions in a convenient format in the neuron object
        return X, Y, Z

    def get_subdivisions(self):
        """
        Get the subdivisions of the synapses.
        """
        return self.subdivisions

    def get_body_id(self):
        """
        Get the body id of the neuron.
        """
        return self.body_id

    def get_synapse_count(self, to=None):
        """
        Get the number of synapses for the neuron.

        Parameters
        ----------
        to : int, optional
            The body id of the postsynaptic neuron.
            If None, the total number of synapses is returned.
            The default is None.
        """
        if to is None:
            return len(self.synapse_df)
        else:
            return len(self.synapse_df[self.synapse_df["end_bid"] == to])

    # --- setters
    def add_neuropil_information(self):
        """
        Add neuropil information to the synapse dataframe.
        """
        self.__categorical_neuropil_information()
        return

    def create_synapse_groups(self, attribute: str):
        """
        Create synapse groups based on an attribute.
        This will be used to split neurons in the Connections class.
        The table is filtered such that connections from a neuron
        to another are removed if the total number of synapses is below
        the threshold in the params file.
        """
        if attribute == "neuropil":
            self.__categorical_neuropil_information()
        if attribute not in self.synapse_df.columns:
            raise AttributeError(f"Attribute {attribute} not in synapse dataframe.")
        # create a connections_df table involving the Neuron.
        # It has two columns 'id_pre' and 'id_post' with the bodyId of the pre and post neurons
        # and a columns 'subdivision_pre'  with the index associated to the attribute split on.
        # Unclassified synapses have a -1 index.
        # Finally, it has a 'synapse_id' column with a list of the synapse ids.
        synapses = self.synapse_df[[attribute, "synapse_id", "start_bid", "end_bid"]]
        # ensure the 'attribute' column is a string
        synapses[attribute] = synapses[attribute].astype(str).values
        # find end_bid for which the number of synapses is below the threshold
        syn_count = synapses.groupby("end_bid").size().reset_index()
        syn_count.columns = ["end_bid", "syn_count"]
        syn_count = syn_count[syn_count["syn_count"] < params.SYNAPSE_CUTOFF]
        to_discard = syn_count["end_bid"].values
        synapses = synapses[~synapses["end_bid"].isin(to_discard)]

        # complete the table
        synapses.fillna(
            {attribute: -1}, inplace=True
        )  # unclassified synapses get together
        # define the subdivision a synapse belongs to by mapping the attribute to an index
        to_map = np.sort(synapses[attribute].dropna().unique()).tolist()
        if "-1" in to_map:  # push unclassified synapses to the end
            to_map.remove("-1")
            to_map.append("-1")

        mapping = {val: i for i, val in enumerate(to_map)}
        mapping[np.nan] = -1 if "-1" not in to_map else mapping["-1"]
        synapses["subdivision_start"] = synapses[attribute].map(mapping)
        synapses["subdivision_start_name"] = synapses[attribute]
        synapses.drop(columns=[attribute], inplace=True)
        synapses = (
            synapses.groupby(
                ["start_bid", "subdivision_start", "subdivision_start_name", "end_bid"]
            )
            .agg(list)
            .reset_index()
        )
        synapses["syn_count"] = synapses["synapse_id"].apply(len)

        self.subdivisions = synapses

    def clear_not_connected(self, not_connected: list[BodyId] | list[int]):
        """
        Clear the subdivisions table from neurons that have their bodyid in
        the not_connected list.
        """
        assert (
            self.subdivisions is not None
        ), "Trying to clear not_connected but subdivisions is None"
        self.subdivisions = self.subdivisions[
            ~self.subdivisions["end_bid"].isin(not_connected)
        ]
        self.synapse_df = self.synapse_df[
            ~self.synapse_df["end_bid"].isin(not_connected)
        ]

    def remove_defined_subdivisions(self):
        """
        Remove the defined subdivisions.
        """
        self.subdivisions = None

    # --- computations
    def cluster_synapses_spatially(
        self,
        n_clusters: int = 3,
        on_attribute: Optional[dict] = None,
    ):
        """
        Cluster the synapses spatially using K-Means clustering.
        If on_attribute is not None, the clustering is done if the attribute
        given as key contains the value given as value.

        Use case:
        on_attribute = {'neuropil': 'T3'}
        clusters the synapses of a neuron in the neuropils with 'T3'
        in their name, namely LegNp(T3)(L) and LegNp(T3)(R). All other
        synapses are clustered under the index '-1'.
        If multiple attributes are given, the combination logic is 'OR'.
        """
        # get the synapse positions
        X, Y, Z = self.get_synapse_distribution()

        # filter if the attribute is given
        if on_attribute is not None:
            kept_indices = []
            if "neuropil" in on_attribute.keys():
                self.__categorical_neuropil_information()
            for (
                key,
                value,
            ) in on_attribute.items():  # only keep the synapses with the attribute
                synapses = self.synapse_df[self.synapse_df[key].str.contains(value)]
                kept_indices.extend(synapses.index)
            X = X[kept_indices]
            Y = Y[kept_indices]
            Z = Z[kept_indices]

        # cluster the synapses
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(np.array([X, Y, Z]).T)
        self.synapse_df["KMeans_cluster"] = kmeans.predict(
            np.array(
                [
                    self.synapse_df["X"],  # keep the order of the synapses
                    self.synapse_df["Y"],
                    self.synapse_df["Z"],
                ]
            ).T
        )

        if on_attribute is not None:  # set the cluster to -1 for the synapses not kept
            self.synapse_df.loc[
                ~self.synapse_df.index.isin(kept_indices), "KMeans_cluster"
            ] = -1
        return

    # --- visualisation
    def plot_synapse_distribution(
        self,
        color_by: Optional[str] = None,
        discrete_coloring: bool = True,
        threshold: bool = False,
        cmap: str | matplotlib.colors.Colormap | typing.Any = params.colorblind_palette,
        ax: Optional[matplotlib.axes.Axes] = None,
        savefig: bool = True,
    ):
        """
        Plot the synapse distribution for the neuron.
        """
        plt.get_cmap()
        X, Y, Z = self.get_synapse_distribution(threshold=threshold)
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=params.FIGSIZE, dpi=params.DPI)
        assert ax is not None  # needed for type hinting
        if color_by is None:
            plots_design.scatter_xyz_2d(X, Y, Z=Z, ax=ax, cmap=cmap)
        else:
            if color_by not in self.synapse_df.columns:
                raise AttributeError(f"Attribute {color_by} not in synapse dataframe.")
            # map self.synapse_df[color_by] categorical values to integers
            if threshold:
                syn_count = self.synapse_df.groupby("end_bid").size().reset_index()
                syn_count.columns = ["end_bid", "syn_count"]
                syn_count = syn_count[syn_count["syn_count"] >= params.SYNAPSE_CUTOFF]
                to_keep = syn_count["end_bid"].values
                synapses = self.synapse_df[self.synapse_df["end_bid"].isin(to_keep)]
            else:
                synapses = self.synapse_df

            Z = synapses[color_by].dropna().values
            plots_design.scatter_xyz_2d(
                X,
                Y,
                Z=Z,
                z_label=color_by,
                ax=ax,
                cmap=cmap,
                discrete_coloring=discrete_coloring,
            )
        if savefig:
            name = os.path.join(
                params.PLOT_DIR,
                f"synapse_distribution_{self.body_id}_color_by_{color_by}.pdf",
                )
            plt.savefig(
                name,
            )
        return ax

    # --- loading and saving
    def save(self, name: str):
        """
        Save the neuron to a file.

        Parameters
        ----------
        name : str
            The name of the file to save to.
        """
        with open(os.path.join(params.NEURON_DIR, name + ".txt"), "wb") as file:
            pickle.dump(self.__dict__, file)


# --- helper functions

@typing.overload
def split_neuron_by_neuropil(
        neuron_id,
        save: bool = True,
        return_type: typing.Literal["Neuron", "name"] = "Neuron"
        ) -> Neuron:...
@typing.overload
def split_neuron_by_neuropil(
    neuron_id,
    save: bool = True,
    return_type: typing.Literal["Neuron", "name"] = "name"
    ) -> str:...

def split_neuron_by_neuropil(
        neuron_id,
        save: bool = True,
        return_type: typing.Literal["Neuron", "name"] = "name"
        ):
    """
    Define neuron subdivisions based on synapse distribution.
    Saves the subdivisions in a new neuron to file, which can be loaded by its
    name.
    """
    # TODO: update the saving and loading directory management as a function of CR
    name = "neuron-" + str(neuron_id) + "_neuropil-split"
    # check there are files starting with name
    #files = [f for f in os.listdir(params.NEURON_DIR) if f.startswith(name)]
    #if files:
    #    return name
    #else:
    neuron = Neuron(neuron_id)
    _ = neuron.get_synapse_distribution(threshold=True)
    neuron.create_synapse_groups(attribute="neuropil")
    if save:
        neuron.save(name=name)
    if return_type == "Neuron":
        return neuron
    else:
        return name
