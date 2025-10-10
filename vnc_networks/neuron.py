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
import polars as pl
from sklearn.cluster import KMeans

from . import params
from .connectome_reader import ConnectomeReader, default_connectome_reader
from .params import BodyId, NeuronAttribute
from .utils import plots_design


class Neuron:
    body_id: BodyId

    def __deepcopy__(self, memo):
        """
        Deepcopy the neuron.
        Only the CR is not deep copied, just referenced.
        """
        new_instance = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_instance

        for k, v in self.__dict__.items():
            if k == "CR":
                setattr(new_instance, k, v)
            else:
                setattr(new_instance, k, copy.deepcopy(v, memo))

        return new_instance

    @typing.overload
    def __init__(
        self,
        from_file: str,
        CR: ConnectomeReader | None = None,
    ): ...
    @typing.overload
    def __init__(
        self,
        body_id: BodyId | int,
        CR: ConnectomeReader | None = None,
    ): ...

    def __init__(
        self,
        body_id: Optional[BodyId | int] = None,
        CR: ConnectomeReader | None = None,
        from_file: Optional[str] = None,
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
        bodyId : int, optional
            The body id of the neuron.
            The default is None.
        from_file : str, optional
            The name of the file to load the neuron from.
            The default is None.
        """
        self.CR = CR or default_connectome_reader()  # will not be saved to file
        self.connectome_name = self.CR.connectome_name
        self.connectome_version = self.CR.connectome_version

        if from_file is not None:
            self.__load(from_file)
        else:
            assert body_id is not None, (
                "To initialise a `Neuron`, you must provide either a `body_id` or `from_file`, but both were None."
            )
            self.body_id = body_id  # type: ignore
            self.data = self.CR.load_data_neuron(
                body_id, self.CR.node_base_attributes()
            )
            self.__initialise_base_attributes()

        # verify that the neuron has the required information (up to date)
        assert hasattr(self, "CR"), "CR not found in neuron."

    # private methods

    def __load(self, name: str):
        """
        Initialise the neuron from a file.
        """
        with open(
            os.path.join(self.CR.get_neuron_save_dir(), name + ".txt"), "rb"
        ) as file:
            neuron = pickle.load(file)
        for key, value in neuron.items():
            if key == "connectome_name":
                assert value == self.CR.connectome_name, (
                    "file created with another connectome!"
                )
            if key == "connectome_version":
                assert value == self.CR.connectome_version, (
                    "file created with another connectome version!"
                )
            setattr(self, key, value)

    def __initialise_base_attributes(self):
        self.nt_type = self.data[0, "nt_type"]
        self.nt_proba = self.data[0, "nt_proba"]
        self.class_1 = self.data[0, "class_1"]
        self.class_2 = self.data[0, "class_2"]
        self.name = self.data[0, "name"]
        self.side = self.data[0, "side"]
        self.neuropil = self.data[0, "neuropil"]
        self.size = self.data[0, "size"]
        self.hemilineage = self.data[0, "hemilineage"]

    def __load_synapse_ids(self):
        # Independently of the file structure, we should get a pl.dataframe
        # with columns
        # ['synapse_id', 'start_bid', 'end_bid', 'X', 'Y', 'Z']

        if "synapse_df" in self.__dict__:  # already done
            return

        self.synapse_df = self.CR.get_synapse_df(self.body_id)

    def __categorical_neuropil_information(self):
        """
        Add a 'neuropil' column to the synapse df.

        Parameters
        ----------
        subset : list, optional
            The subset of synapse ids to convert.
            The default is None, which converts all synapse ids.
        """
        if "synapse_df" not in self.__dict__:
            self.__load_synapse_ids()

        if "neuropil" in self.synapse_df.columns:  # already done
            return

        data = self.CR.get_synapse_neuropil(
            synapse_ids=self.synapse_df["synapse_id"].to_list(),
        )

        # replace None values with 'None'
        data = data.with_columns(neuropil=pl.col("neuropil").fill_null("None"))

        # merge with existing synapse df
        self.synapse_df = self.synapse_df.join(data, on="synapse_id", how="inner")
        return

    # public methods
    # --- getters
    def get_data(self):
        return self.data

    def get_synapse_table(self):
        """
        Get the synapse table of the connections.
        """
        if "synapse_df" not in self.__dict__:
            self.__load_synapse_ids()
        return copy.deepcopy(self.synapse_df)

    def get_synapse_distribution(
        self,
        threshold: bool = False,
        z_attribute: Optional[str] = None,
        angle: int = 0,
    ):
        """
        Get the synapse distribution for the neuron.

        Parameters
        ----------
        threshold : bool, optional
            Whether to apply a threshold to the synapse distribution.
            The default is False.
        z_attribute : str, optional
            The attribute to use for the z-axis.
            The default is None, returning the Z coordinate.
        angle : int, optional
            The angle in degrees to rotate the synapse distribution along the
            x-axis, i.e. Y and Z are modified.
            The default is 0.
        """
        # check if the synapse df is loaded
        self.__load_synapse_ids()

        # threshold if for a given neuron, the number of synapses is below the threshold
        if threshold:
            # syn_count = self.synapse_df.groupby("end_bid").size().reset_index()
            # syn_count.columns = ["end_bid", "syn_count"]
            # syn_count = syn_count[syn_count["syn_count"] >= params.SYNAPSE_CUTOFF]
            # to_keep = syn_count["end_bid"].values
            # synapses = self.synapse_df[self.synapse_df["end_bid"].isin(to_keep)]
            synapses = self.synapse_df.filter(
                pl.len().over("end_bid") >= params.SYNAPSE_CUTOFF
            )
        else:
            synapses = self.synapse_df

        if z_attribute is not None and z_attribute not in self.synapse_df.columns:
            raise AttributeError(f"Attribute {z_attribute} not in synapse dataframe.")

        # get the synapse positions
        angle = np.radians(angle)
        X = synapses["X"]
        Y = np.cos(angle) * synapses["Y"] - np.sin(angle) * synapses["Z"]
        if z_attribute is None:
            Z = np.sin(angle) * synapses["Y"] + np.cos(angle) * synapses["Z"]
        else:
            Z = synapses[z_attribute]

        # save the positions in a convenient format in the neuron object
        return X, Y, Z

    def get_subdivisions(self):
        """
        Get the subdivisions of the synapses.
        """
        return self.subdivisions

    def get_body_id(self) -> BodyId:
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
            return len(self.synapse_df.filter(end_bid=to))

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

        synapses = (
            self.synapse_df[[attribute, "synapse_id", "start_bid", "end_bid"]]
            .filter(pl.len().over("end_bid") >= params.SYNAPSE_CUTOFF)
            .with_columns(pl.col(attribute).fill_null(-1))
        )
        # define the subdivision a synapse belongs to by mapping the attribute to an index
        to_map = synapses[attribute].unique().sort().to_list()
        if "-1" in to_map:  # push unclassified synapses to the end
            to_map.remove("-1")
            to_map.append("-1")
        mapping = {val: i for i, val in enumerate(to_map)}

        synapses = (
            synapses.with_columns(
                subdivision_start=pl.col(attribute)
                .replace(
                    pl.Series(mapping.keys(), strict=False),
                    pl.Series(mapping.values(), strict=False),
                )
                .cast(pl.Int32),  # subdivision_start should be an int
                subdivision_start_name=pl.col(attribute),
            )
            .with_columns(
                syn_count=pl.len().over(
                    "start_bid", "end_bid", "subdivision_start_name", "end_bid"
                ),
            )
            .drop(attribute)
        )

        self.subdivisions = synapses

    def clear_not_connected(self, not_connected: list[BodyId] | list[int]):
        """
        Clear the subdivisions table from neurons that have their bodyid in
        the not_connected list.
        """
        assert self.subdivisions is not None, (
            "Trying to clear not_connected but subdivisions is None"
        )
        # FIX: should this be start_bid in the first one?
        self.subdivisions = self.subdivisions.filter(
            ~pl.col("end_bid").is_in(not_connected)
            & ~pl.col("end_bid").is_in(not_connected)
        )

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
                kept_indices.extend(
                    self.synapse_df.with_row_index().filter(
                        pl.col(key).str.contains(value)
                    )["index"]
                )
            X = X[kept_indices]
            Y = Y[kept_indices]
            Z = Z[kept_indices]

        # cluster the synapses
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(np.array([X, Y, Z]).T)
        self.synapse_df.with_columns(
            KMeans_cluster=kmeans.predict(
                np.array(
                    [
                        self.synapse_df["X"],  # keep the order of the synapses
                        self.synapse_df["Y"],
                        self.synapse_df["Z"],
                    ]
                ).T
            ).tolist()
        )

        if on_attribute is not None:  # set the cluster to -1 for the synapses not kept
            self.synapse_df = (
                self.synapse_df.with_row_index()
                .with_columns(
                    KMeans_cluster=pl.when(pl.col("index").is_in(kept_indices))
                    .then(pl.col("KMeans_cluster"))
                    .otherwise(-1)
                )
                .drop("index")
            )

    # --- visualisation
    def plot_synapse_distribution(
        self,
        color_by: Optional[str] = None,
        discrete_coloring: bool = True,
        threshold: bool = False,
        cmap: str | matplotlib.colors.Colormap | typing.Any = params.colorblind_palette,
        ax: Optional[matplotlib.axes.Axes] = None,
        savefig: bool = True,
        angle: int = 0,
    ):
        """
        Plot the synapse distribution for the neuron.
        """
        plt.get_cmap()
        X, Y, Z = self.get_synapse_distribution(threshold=threshold, angle=angle)
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
                synapses = self.synapse_df.filter(
                    pl.len().over("end_bid") >= params.SYNAPSE_CUTOFF
                )
            else:
                synapses = self.synapse_df

            Z = synapses[color_by]
            # Z = synapses[color_by].dropna().values
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
                self.CR.get_plots_dir(),
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
        # save __dict__ except for the CR attribute
        to_save = {key: value for key, value in self.__dict__.items() if key != "CR"}
        with open(
            os.path.join(self.CR.get_neuron_save_dir(), name + ".txt"), "wb"
        ) as file:
            pickle.dump(to_save, file)


# --- helper functions


@typing.overload
def split_neuron_by_neuropil(
    neuron_id,
    CR: ConnectomeReader | None = None,
    *,
    save: bool = True,
    return_type: typing.Literal["name"] = "name",
) -> str: ...
@typing.overload
def split_neuron_by_neuropil(
    neuron_id,
    CR: ConnectomeReader | None = None,
    *,
    save: bool = True,
    return_type: typing.Literal["Neuron"],
) -> Neuron: ...


def split_neuron_by_neuropil(
    neuron_id,
    CR: ConnectomeReader | None = None,
    *,
    save: bool = True,
    return_type: typing.Literal["Neuron", "name"] = "name",
):
    """
    Define neuron subdivisions based on synapse distribution.
    Saves the subdivisions in a new neuron to file, which can be loaded by its
    name.
    """
    name = "neuron-" + str(neuron_id) + "_neuropil-split"
    # check there are files starting with name
    # files = [f for f in os.listdir(params.NEURON_DIR) if f.startswith(name)]
    # if files:
    #    return name
    # else:
    neuron = Neuron(neuron_id, CR=CR)
    _ = neuron.get_synapse_distribution(threshold=True)
    neuron.create_synapse_groups(attribute="neuropil")
    if save:
        neuron.save(name=name)
    if return_type == "Neuron":
        return neuron
    else:
        return name
