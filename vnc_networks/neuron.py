"""
Initialises the Neuron class.
Meant to be used to load data for a single neuron.
Possible use cases are visualisation, or looking at synapse distributions.

Possible upgrade: possible to match synapses on the pre and post neurons
by using the dataset "Neuprint_Synapse_Connections_manc_v1.ftr" which
has two columns: ':START_ID(Syn-ID)' and ':END_ID(Syn-ID)'.
"""

import os
import pickle
import typing
from typing import Optional

import matplotlib.axes
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import params
import utils.plots_design as plot_design
from get_nodes_data import load_data_neuron
from params import BodyId, NeuronAttribute
from sklearn.cluster import KMeans

NEURON_BASE_ATTRIBUTES: list[NeuronAttribute] = [
    "systematicType:string",
    "hemilineage:string",
    "somaSide:string",
    "class:string",
    "subclass:string",
    "group:int",
    "cellBodyFiber:string",
    "size:long",
    "target:string",
    "predictedNtProb:float",
    "predictedNt:string",
    "tag:string",
    "status:string",  # 'Traced' or None
]


class Neuron:
    def __init__(
        self, bodyId: Optional[BodyId | int] = None, from_file: Optional[str] = None
    ):
        """
        Initialise the neuron.
        Loading possible either from scratch using only the bodyId, or
        from a file. Loading from a file is useful if the neuron has already
        been processed and saved. Loading from scratch is useful if the neuron
        is new, but is computationally expensive and time-consuming.

        Parameters
        ----------
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
                bodyId is not None
            ), "To initialise a `Neuron`, you must provide either a `bodyId` or `from_file`, but both were None."
            self.bodyId = bodyId
            self.data = load_data_neuron(bodyId, NEURON_BASE_ATTRIBUTES)
            self.__initialise_base_attributes()

    # private methods

    def __load(self, name: str):
        """
        Initialise the neuron from a file.
        """
        with open(os.path.join(params.NEURON_DIR, name + ".txt"), "rb") as file:
            neuron = pickle.load(file)
        self.__dict__.update(neuron)

    def __initialise_base_attributes(self):
        self.type = self.data["systematicType:string"].values[0]
        self.hemilineage = self.data["hemilineage:string"].values[0]
        self.soma_side = self.data["somaSide:string"].values[0]
        self.class_ = self.data["class:string"].values[0]
        self.subclass = self.data["subclass:string"].values[0]
        self.group = self.data["group:int"].values[0]
        self.cell_body_fiber = self.data["cellBodyFiber:string"].values[0]
        self.size = self.data["size:long"].values[0]
        self.target = self.data["target:string"].values[0]
        self.predicted_nt_prob = self.data["predictedNtProb:float"].values[0]
        self.predicted_nt = self.data["predictedNt:string"].values[0]
        self.tag = self.data["tag:string"].values[0]

    def __load_synapse_ids(self):
        """
        Load the synapse ids for the neuron.
        """
        # neuron to synapse set
        neuron_to_synapse = pd.read_feather(params.NEUPRINT_NEURON_SYNAPSESET_FILE)
        synset_list = neuron_to_synapse.loc[
            neuron_to_synapse[":START_ID(Body-ID)"] == self.bodyId
        ][":END_ID(SynSet-ID)"].values

        # synapse set to synapse
        synapses = pd.read_feather(params.NEUPRINT_SYNAPSESET_FILE)
        synapses = synapses.loc[synapses[":START_ID(SynSet-ID)"].isin(synset_list)]
        synapses.reset_index(drop=True, inplace=True)

        # build a dataframe with columns 'syn_id', 'synset_id'
        synapse_df = pd.DataFrame(
            {
                "syn_id": synapses[":END_ID(Syn-ID)"],
                "synset_id": synapses[":START_ID(SynSet-ID)"],
            }
        )
        synapse_df["start_id"] = synapse_df["synset_id"].apply(
            lambda x: int(x.split("_")[0])
        )  # body id of the presynaptic neuron
        synapse_df["end_id"] = synapse_df["synset_id"].apply(
            lambda x: int(x.split("_")[1])
        )  # body id of the postsynaptic neuron
        synapse_df["position"] = synapse_df["synset_id"].apply(
            lambda x: x.split("_")[2]
        )  # pre or post

        # remove the synapses that belong to partner neurons
        synapse_df = synapse_df[synapse_df["position"] == "pre"]
        synapse_df.drop(columns=["position"], inplace=True)

        # set the synapse ids
        self.synapse_df = synapse_df
        del synapses, neuron_to_synapse
        return

    def __load_synapse_locations(self):
        """
        Convert the synapse ids to MANC format.

        Parameters
        ----------
        subset : list, optional
            The subset of synapse ids to convert.
            The default is None, which converts all synapse ids.
        """

        # load the synapse data
        data = pd.read_feather(
            params.NEUPRINT_SYNAPSE_FILE,
            columns=[":ID(Syn-ID)", "location:point{srid:9157}"],
        )
        data = data.loc[data[":ID(Syn-ID)"].isin(self.synapse_df["syn_id"])]

        # merge with existing synapse df
        self.synapse_df = self.synapse_df.merge(
            data, right_on=":ID(Syn-ID)", left_on="syn_id", how="inner"
        )
        del data
        return

    def __categorical_neuropil_information(self):
        """
        Add a column 'neuropil' to the synapse df that contains the neuropil
        in which the synapse is located. This is done by finding the roi column
        that has a bool value of True.
        """
        if "neuropil" in self.synapse_df.columns:  # already done
            return

        roi_file = os.path.join(params.NEUPRINT_RAW_DIR, "all_ROIs.txt")
        rois = list(pd.read_csv(roi_file, sep="\t").values.flatten())

        # find the subset of the synapses in the dataset that we care about for this neuron
        # then for each possible ROI, check which synapses are in that ROI
        # store each synapse's ROI in the neuropil column
        synapses_we_care_about = pd.read_feather(
            params.NEUPRINT_SYNAPSE_FILE, columns=[":ID(Syn-ID)"]
        )[":ID(Syn-ID)"].isin(self.synapse_df["syn_id"])
        self.synapse_df["neuropil"] = "None"
        for roi in rois:
            column_name = roi + ":boolean"
            roi_column = pd.read_feather(
                params.NEUPRINT_SYNAPSE_FILE,
                columns=[column_name, ":ID(Syn-ID)"],
            )[synapses_we_care_about]
            synapses_in_roi = roi_column.loc[
                roi_column[column_name], ":ID(Syn-ID)"
            ].values  # type: ignore
            self.synapse_df.loc[
                self.synapse_df["syn_id"].isin(synapses_in_roi), "neuropil"
            ] = roi

    def __explicit_synapse_positions(self):
        """
        convert the synapse locations defined in the text version of a dict
        to explicit position columns in the synapse df.
        """
        if "X" in self.synapse_df.columns:  # already done
            return

        locations = self.synapse_df["location:point{srid:9157}"].values
        X, Y, Z = [], [], []
        for loc in locations:
            pos = eval(loc)  # read loc as a dict, use of x,y,z under the hood
            if not isinstance(pos, dict):
                pos = {0: np.nan, 1: np.nan, 2: np.nan}
            X.append(pos[0])
            Y.append(pos[1])
            Z.append(pos[2])
        self.synapse_df["X"] = X
        self.synapse_df["Y"] = Z
        self.synapse_df["Z"] = Y
        return

    # public methods
    # --- getters
    def get_data(self):
        return self.data

    def get_synapse_table(self):
        """
        Get the synapse table of the connections.
        """
        synapses = self.synapse_df
        return synapses

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
        try:
            self.synapse_df
        except AttributeError:
            print("Synapse data not loaded. Loading now...")
            self.__load_synapse_ids()
            self.__load_synapse_locations()
            print("... Synapse data loaded.")

        # define synapse positions
        self.__explicit_synapse_positions()
        # threshold if for a given neuron, the number of synapses is below the threshold
        if threshold:
            syn_count = self.synapse_df.groupby("end_id").size().reset_index()
            syn_count.columns = ["end_id", "syn_count"]
            syn_count = syn_count[syn_count["syn_count"] >= params.SYNAPSE_CUTOFF]
            to_keep = syn_count["end_id"].values
            synapses = self.synapse_df[self.synapse_df["end_id"].isin(to_keep)]
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
        return self.bodyId

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
            return len(self.synapse_df[self.synapse_df["end_id"] == to])

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
        # Finally, it has a 'synapse_ids' column with a list of the synapse ids.
        synapses = self.synapse_df[[attribute, "syn_id", "start_id", "end_id"]]
        # ensure the 'attribute' column is a string
        synapses[attribute] = synapses[attribute].astype(str).values
        # find end_id for which the number of synapses is below the threshold
        syn_count = synapses.groupby("end_id").size().reset_index()
        syn_count.columns = ["end_id", "syn_count"]
        syn_count = syn_count[syn_count["syn_count"] < params.SYNAPSE_CUTOFF]
        to_discard = syn_count["end_id"].values
        synapses = synapses[~synapses["end_id"].isin(to_discard)]

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
        print(mapping)
        synapses["subdivision_start"] = synapses[attribute].map(mapping)
        synapses["subdivision_start_name"] = synapses[attribute]
        synapses.drop(columns=[attribute], inplace=True)
        synapses = (
            synapses.groupby(
                ["start_id", "subdivision_start", "subdivision_start_name", "end_id"]
            )
            .agg(list)
            .reset_index()
        )
        synapses["syn_count"] = synapses["syn_id"].apply(len)

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
            ~self.subdivisions["end_id"].isin(not_connected)
        ]
        self.synapse_df = self.synapse_df[
            ~self.synapse_df["end_id"].isin(not_connected)
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
            fig, ax = plt.subplots(1, 1, figsize=params.FIGSIZE, dpi=params.DPI)[1]
        assert ax is not None  # needed for type hinting
        if color_by is None:
            plot_design.scatter_xyz_2d(X, Y, Z=Z, ax=ax, cmap=cmap)
        else:
            if color_by not in self.synapse_df.columns:
                raise AttributeError(f"Attribute {color_by} not in synapse dataframe.")
            # map self.synapse_df[color_by] categorical values to integers
            if threshold:
                syn_count = self.synapse_df.groupby("end_id").size().reset_index()
                syn_count.columns = ["end_id", "syn_count"]
                syn_count = syn_count[syn_count["syn_count"] >= params.SYNAPSE_CUTOFF]
                to_keep = syn_count["end_id"].values
                synapses = self.synapse_df[self.synapse_df["end_id"].isin(to_keep)]
            else:
                synapses = self.synapse_df

            Z = synapses[color_by].dropna().values
            plot_design.scatter_xyz_2d(
                X,
                Y,
                Z=Z,
                z_label=color_by,
                ax=ax,
                cmap=cmap,
                discrete_coloring=discrete_coloring,
            )
        if savefig:
            plt.savefig(
                f"{params.PLOT_DIR}/synapse_distribution_{self.bodyId}_{color_by}.pdf"
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


def split_neuron_by_neuropil(neuron_id):
    """
    Define neuron subdivisions based on synapse distribution.
    Saves the subdivisions in a new neuron to file, which can be loaded by its
    name.
    """
    name = "neuron-" + str(neuron_id) + "_neuropil-split"
    # check there are files starting with name
    files = [f for f in os.listdir(params.NEURON_DIR) if f.startswith(name)]
    if files:
        return name
    else:
        neuron = Neuron(neuron_id)
        _ = neuron.get_synapse_distribution(threshold=True)
        neuron.create_synapse_groups(attribute="neuropil")
        neuron.save(name=name)
    return name
