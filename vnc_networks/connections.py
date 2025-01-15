"""
Initializes the connections class.
Possible use cases are processing the adjacency matrix from a set of neurons,
or displaying the connections between neurons in a graph.
Useful class when juggling between different representations of the connectome.

Each neuron is referenced by a tuple of (bodyId, subdivision) where the
subdivision is a number indexing a set of synapses from the same neuron. This
allows to treat a single neuron as multiple nodes in the graph.
Each such created neuron has a unique identifier associated (uid).

Use the following code to initialize the class:
```
from connections import Connections
neurons_pre = get_neurons_from_class('sensory neuron')
neurons_post = get_neurons_from_class('motor neuron')
connections = Connections(neurons_pre, `neurons_post`)
```
"""
import copy
import os
import pickle
import typing
from collections.abc import Mapping
from typing import Optional

import cmatrix
import get_nodes_data
import matplotlib.axes
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import params
import seaborn as sns
import utils.matrix_utils as matrix_utils
import utils.nx_design as nx_design
import utils.nx_utils as nx_utils
import utils.plots_design as plots_design
from neuron import Neuron
from params import UID, BodyId, NeuronAttribute, SelectionDict

## ---- Types ---- ##
SortingStyle = typing.Literal[
    "degree",
    "betweenness",
    "from_index",
    "input_clustering",
    "centrality",
    "output_clustering",
]

## ---- Constants ---- ##
FIGSIZE = (8, 8)
DPI = 300

## ---- Classes ---- ##


class Connections:
    def __init__(
        self,
        from_file: Optional[str] = None,
        neurons_pre: Optional[list[int] | list[BodyId]] = None,
        neurons_post: Optional[list[int] | list[BodyId]] = None,
        nt_weights: Optional[Mapping[str, int]] = params.NT_WEIGHTS,
        split_neurons: Optional[list[Neuron]] = None,
        not_connected: Optional[list[BodyId] | list[int]] = None,
        keep_only_traced_neurons: Optional[bool] = True,
    ):
        """
        If no neurons_post are given, the class assumes the same neurons as pre-synaptic.
        If no neurons_pre are given, the class considers all connections defined in the database.

        If not used as a standalone class, the initialize() method should be called after instantiation.
        """
        if from_file is not None:
            self.__load(from_file)
        else:
            if neurons_pre is None:
                neurons_ = pd.read_feather(
                    params.NEUPRINT_NODES_FILE, columns=[":ID(Body-ID)"]
                )
                neurons_pre_ = neurons_[":ID(Body-ID)"].to_list()

            else:
                neurons_pre_ = neurons_pre
            self.neurons_pre = neurons_pre_
            if neurons_post is None:
                self.neurons_post = self.neurons_pre
            else:
                self.neurons_post = (neurons_post,)
            self.nt_weights = nt_weights
            self.subgraphs = {}

        self.__initialize(split_neurons, not_connected)

        if keep_only_traced_neurons: 
            # this is computationally not optimal but makes the dataset a lot cleaner.
            # TODO: implement this as the dataset is created, by loading 
            # node data seuentially.
            self.get_connections_with_only_traced_neurons()

    # private methods
    def __initialize(
        self,
        split_neurons: Optional[list[Neuron]] = None,
        not_connected: Optional[list[BodyId] | list[int]] = None,  # body ids
    ):
        if split_neurons is not None and not_connected is not None:
            for neuron in split_neurons:
                neuron.clear_not_connected(not_connected)
        self.__get_connections()
        self.__remove_connections_between(not_connected=not_connected)
        self.__compute_effective_weights_in_connections()
        self.__split_neurons_in_connections(split_neurons)
        self.__map_uid()
        self.__name_neurons(split_neurons)
        self.__build_graph()
        self.__build_adjacency_matrices()
        return

    def __load(self, name: str):
        """
        Import the object from a pickle file.
        """
        filename = os.path.join(params.CONNECTION_DIR, name + ".txt")
        with open(filename, "rb") as file:
            neuron = pickle.load(file)
        self.__dict__.update(neuron)

    def __get_connections(self):
        connections_ = pd.read_feather(
            params.NEUPRINT_CONNECTIONS_FILE,
            columns=[":START_ID(Body-ID)", "weightHR:int", ":END_ID(Body-ID)"],
        )
        # filter out only the connections relevant here
        connections_ = connections_[
            connections_[":START_ID(Body-ID)"].isin(self.neurons_pre)
            & connections_[":END_ID(Body-ID)"].isin(self.neurons_post)
        ]

        # rename synapse count column explicitly and filter
        connections_["syn_count"] = connections_["weightHR:int"]
        connections_ = connections_[connections_["syn_count"] >= params.SYNAPSE_CUTOFF]

        # add relevant information for the processing steps

        ## add the neurotransmitter type to the connections
        nttypes = get_nodes_data.load_data_neuron_set(
            self.neurons_pre,
            ["predictedNt:string"],
        )
        connections_ = pd.merge(
            connections_,
            nttypes,
            left_on=":START_ID(Body-ID)",
            right_on=":ID(Body-ID)",
            how="left",
            suffixes=("", ""),
        )

        connections_ = connections_.drop(columns=[":ID(Body-ID)"])
        weight_vec = np.array(
            [self.nt_weights[x] for x in connections_["predictedNt:string"]]
        )
        connections_["eff_weight"] = connections_["syn_count"] * weight_vec

        # add a column with 'subdivision_start' and 'subdivision_end' with zeros as values
        connections_["subdivision_start"] = 0
        connections_["subdivision_end"] = 0

        self.connections = connections_

    def __compute_effective_weights_in_connections(self, ids: str = "body_id"):
        """
        Compute the effective weights of the connections.
        """
        if ids == "body_id":
            grouping_on = ":END_ID(Body-ID)"
        elif ids == "uid":
            grouping_on = "end_uid"
        else:
            raise ValueError(
                f"Class Connections \
                ::: > compute_effective_weights_in_connections():\
                Unknown id type {ids}"
            )
        ## Calculate effective weights normalized by total incoming synapses
        in_total = self.connections.groupby(grouping_on)["syn_count"].sum()
        in_total = in_total.to_frame(name="in_total")
        self.connections = self.connections.merge(
            in_total, left_on=grouping_on, right_index=True
        )
        self.connections["syn_count_norm"] = (
            self.connections["syn_count"] / self.connections["in_total"]
        )
        self.connections["eff_weight_norm"] = (
            self.connections["eff_weight"] / self.connections["in_total"]
        )
        self.connections = self.connections.drop(columns=["in_total"])
        return

    def __remove_connections_between(
        self, not_connected: Optional[list[BodyId] | list[int]] = None
    ):
        """
        Remove connections between neurons in the not_connected list.

        Parameters
        ----------
        not_connected: list[int]
            List of body ids that should not be connected.
        """
        if not_connected is None:
            return
        self.connections = self.connections[
            ~(
                (self.connections[":START_ID(Body-ID)"].isin(not_connected))
                & (self.connections[":END_ID(Body-ID)"].isin(not_connected))
            )
        ]

    def __split_neuron(self, neuron: Neuron):
        """
        Split a single neuron into multiple nodes.
        """
        # verify that there is a subdivision defined
        subdivisions = (
            neuron.get_subdivisions()
        )  # table with connections split by subdivision
        assert subdivisions is not None, "Neuron subdivisions is None"
        split_body_id = neuron.get_body_id()

        split_neuron_subdivision_ids = subdivisions["subdivision_start"].unique()

        # --- Manage the splitting as pre-synaptic neuron
        target_neurons = subdivisions["end_id"].unique()  # body ids
        for end_body_id in target_neurons:
            # get the connections to split
            subdivisions_ = subdivisions[
                (subdivisions["start_id"] == split_body_id)
                & (subdivisions["end_id"] == end_body_id)
            ]
            n_rows_in_target = len(subdivisions_)
            # sanity check on the total synapse count
            n_syn_in_connections_table = self.get_nb_synapses(
                start_id=split_body_id, end_id=end_body_id, input_type="body_id"
            )
            n_syn_in_neuron = neuron.get_synapse_count(to=end_body_id)
            if not n_syn_in_connections_table == n_syn_in_neuron:
                raise ValueError(
                    f"The total synapse count from {split_body_id} \
                    to {end_body_id} is not consistent: \
                    {n_syn_in_connections_table} in the connections table,\
                    {n_syn_in_neuron} in the neuron model."
                )
            # get the data for the neuron pair before splitting
            template = self.connections[
                (self.connections[":START_ID(Body-ID)"] == split_body_id)
                & (self.connections[":END_ID(Body-ID)"] == end_body_id)
            ].copy()
            # duplicate the template len(subdivisions_) times
            new_connections_ = pd.concat(
                [template] * len(subdivisions_), ignore_index=True
            )  # these can be multiple rows if the target is subdivided
            subdivisions_ = pd.concat(
                [subdivisions_] * len(template), ignore_index=True
            )  # will duplicate the operations to apply to each end subdivision
            # add the subdivisions to the new connections
            new_connections_["subdivision_start"] = subdivisions_[
                "subdivision_start"
            ]  # of the form [0,1,2,0,1,2,...]
            # compute the synaptic ratios for each subdivision
            ratios = np.array(subdivisions_["syn_count"] / n_syn_in_neuron)[
                0:n_rows_in_target
            ]  # original synapse distribution
            # update the synapse count in the new connections
            new_connections_["syn_count"] = [
                int(x * y) for x in template["syn_count"] for y in ratios
            ]  # ! not permutable
            new_connections_["eff_weight"] = [
                x * y for x in template["eff_weight"] for y in ratios
            ]
            new_connections_["syn_count_norm"] = [
                x * y for x in template["syn_count_norm"] for y in ratios
            ]
            new_connections_["eff_weight_norm"] = [
                x * y for x in template["eff_weight_norm"] for y in ratios
            ]
            # replace the template line with the new connections
            self.connections = pd.concat(
                [
                    self.connections[  # remove the initial entry
                        ~(
                            (self.connections[":START_ID(Body-ID)"] == split_body_id)
                            & (self.connections[":END_ID(Body-ID)"] == end_body_id)
                        )
                    ],
                    new_connections_,  # add the new entries
                ],
                ignore_index=True,
            )

        # --- Manage the splitting as post-synaptic neuron
        relevant_inputs = self.connections[
            self.connections[":END_ID(Body-ID)"] == split_body_id
        ][":START_ID(Body-ID)"].unique()
        for start_body_id in relevant_inputs:
            template = self.connections[
                (self.connections[":START_ID(Body-ID)"] == start_body_id)
                & (self.connections[":END_ID(Body-ID)"] == split_body_id)
            ].copy()
            # duplicate the template for each existing subdivision
            new_connections_ = pd.concat(
                [template] * len(split_neuron_subdivision_ids), ignore_index=True
            )
            # add the subdivisions to the new connections
            new_connections_["subdivision_end"] = [
                i
                for i in split_neuron_subdivision_ids
                for _ in range(len(template))  # ! not permutable
            ]  # of the form [0,0,..1,1,..2,2,..]
            # The other values are kept the same
            # (dendrites converge in our model)
            self.connections = pd.concat(
                [
                    self.connections[
                        ~(
                            (self.connections[":START_ID(Body-ID)"] == start_body_id)
                            & (self.connections[":END_ID(Body-ID)"] == split_body_id)
                        )
                    ],
                    new_connections_,
                ],
                ignore_index=True,
            )

    def __split_neurons_in_connections(
        self, split_neurons: Optional[list[Neuron]] = None
    ):
        """
        Divide the neurons that have more than one subdivision into multiple nodes.
        Each subdivision is considered as a separate node, based on the list of
        synapses within.

        *Import design choice*:
        We consider a neuron model where the dendrites integrate information
        across branches, and then the axon branches out to multiple targets.
        This means that when a neuron is subdivided, the splitting is different
        depending on whether the neuron is pre- or post-synaptic.
        - If the split neuron is pre-synaptic, the axon is split into multiple
        branches, each targeting a different post-synaptic neuron. The synaptic
        weights are divided accordingly.
        - If the split neuron is post-synaptic, the dendrites converge, and the
        split neurons receive the same input from the pre-synaptic neuron. In
        practice this means that the inputs get duplicated.
        """
        if split_neurons is None:
            return

        for N_ in split_neurons:
            self.__split_neuron(N_)

        return

    def __map_uid(self):
        """
        Map the tuples (bodyId, subdivision) to a unique identifier (uid) number that will
        be used to reference the newly defined neurons, e.g. as graph node ids.
        """
        # reference all unique tuples in (":START_ID(Body-ID)", "subdivision_start")
        # and (":END_ID(Body-ID)", "subdivision_end") from the self.connections dataframe
        unique_objects = list(
            set(
                zip(
                    self.connections[":START_ID(Body-ID)"],
                    self.connections["subdivision_start"],
                )
            ).union(
                set(
                    zip(
                        self.connections[":END_ID(Body-ID)"],
                        self.connections["subdivision_end"],
                    )
                )
            )
        )
        # create a new df mapping elements in the set to integers
        uids = list(range(len(unique_objects)))
        self.uid = pd.DataFrame(
            {
                "uid": uids,
                "neuron_ids": unique_objects,
                "body_id": [x[0] for x in unique_objects],
                "subdivision": [x[1] for x in unique_objects],
            }
        )
        # add uids to the connections table
        self.connections["start_uid"] = self.__convert_neuron_ids_to_uid(
            self.connections[[":START_ID(Body-ID)", "subdivision_start"]].values,
            input_type="table",
        )
        self.connections["end_uid"] = self.__convert_neuron_ids_to_uid(
            self.connections[[":END_ID(Body-ID)", "subdivision_end"]].values,
            input_type="table",
        )
        return

    def __convert_neuron_ids_to_uid(
        self,
        neuron_ids,
        input_type: str = "tuple",
    ):
        """
        Convert a list of neuron ids to a list of unique identifiers.

        Parameters
        ----------
        neuron_ids: list[tuple[int]] or dataframe['id','subdivision']
            List of neuron ids to convert.
        input_type: str
            Type of the input list.
            Can be 'tuple', 'table'
        """
        if input_type == "tuple":
            tuple_of_ids = neuron_ids
        elif input_type == "table":
            tuple_of_ids = [tuple(row) for row in neuron_ids]
        else:
            raise ValueError(
                f"Class Connections \
                ::: > convert_neuron_ids_to_uid(): Unknown input type {input_type}"
            )
        uid_table = self.uid.copy()
        uid_table = uid_table[uid_table["neuron_ids"].isin(tuple_of_ids)]
        # sort the table by the order of the input list
        uid_table = uid_table.set_index("neuron_ids").loc[tuple_of_ids].reset_index()
        uids = uid_table["uid"].to_list()
        return uids

    @typing.overload
    def __convert_uid_to_neuron_ids(
        self,
        uids,
        output_type: typing.Literal["tuple"] = "tuple",
    ) -> list[tuple[int, int]]: ...
    @typing.overload
    def __convert_uid_to_neuron_ids(
        self,
        uids,
        output_type: typing.Literal["table"] = "table",
    ) -> pd.DataFrame: ...
    @typing.overload
    def __convert_uid_to_neuron_ids(
        self,
        uids,
        output_type: typing.Literal["body_id"] = "body_id",
    ) -> list[BodyId]: ...
    @typing.overload
    def __convert_uid_to_neuron_ids(
        self,
        uids,
        output_type: typing.Literal["subdivision"] = "subdivision",
    ): ...
    def __convert_uid_to_neuron_ids(
        self,
        uids,
        output_type: typing.Literal[
            "tuple", "table", "body_id", "subdivision"
        ] = "tuple",
    ) -> list[tuple[int, int]] | pd.DataFrame | list[BodyId]:
        """
        Convert a list of unique identifiers to a list of neuron ids.

        Parameters
        ----------
        uids: list[int]
            List of unique identifiers to convert.
        output_type: str
            Type of the output list.
            Can be 'tuple', 'table' (dataframe), 'body_id' (only the body ids),
            'subdivision' (only the subdivisions).
        """
        if not isinstance(uids, list):
            uids = [uids]
        if output_type not in ["tuple", "table", "body_id", "subdivision"]:
            raise ValueError(
                f"Class Connections \
                ::: > convert_uid_to_neuron_ids():\
                Unknown output type {output_type}"
            )
        to_return = self.uid.loc[self.uid["uid"].isin(uids)]
        if output_type == "tuple":
            return list(to_return["neuron_ids"].values)
        elif output_type == "table":
            return to_return
        elif output_type == "body_id":
            return list(to_return["body_id"].values)
        elif output_type == "subdivision":
            return to_return["subdivision"].values

    def __get_uids_from_bodyids(self, body_ids: list[BodyId] | list[int]) -> list[UID]:
        """
        Get the unique identifiers from a list of body ids.
        """
        return self.uid.loc[self.uid["body_id"].isin(body_ids)]["uid"].to_list()

    def __build_graph(self):  # networkx graph
        self.graph = nx.from_pandas_edgelist(
            self.connections,
            source="start_uid",
            target="end_uid",
            edge_attr=[
                "syn_count",  # absolute synapse count
                "predictedNt:string",
                "eff_weight",  # signed synapse count (nt weighted)
                "syn_count_norm",  # input normalized synapse count
                "eff_weight_norm",  # input normalized signed synapse count
            ],
            create_using=nx.DiGraph,
        )
        # set an edge attribute 'weight' to the signed synapse count, for compatibility with nx functions
        nx.set_edge_attributes(
            self.graph,
            {
                (u, v): {"weight": d["eff_weight"]}
                for u, v, d in self.graph.edges(data=True)
            },
        )
        # add node attributes
        nx.set_node_attributes(
            self.graph,
            {
                node: body_id
                for node, body_id in zip(self.uid["uid"], self.uid["body_id"])
            },
            "body_id",
        )
        nx.set_node_attributes(
            self.graph,
            {
                node: label
                for node, label in zip(self.uid["uid"], self.uid["node_label"])
            },
            "node_label",
        )
        self.__defined_attributes_in_graph = [
            "body_id",
            "node_label",
        ]
        _ = self.__get_node_attributes("class:string")
        nx.set_node_attributes(
            self.graph, nx.get_node_attributes(self.graph, "class:string"), "node_class"
        )
        self.__defined_attributes_in_graph.append("node_class")

    def __name_neurons(self, split_neurons: Optional[list[Neuron]] = None):
        """
        For the neurons that are split, append the subdivision label
        to the neuron name in the graph.
        For instance, if neuron A is split in A1 and A2, with A1 corresponding
        to synapses in the neuropil T1 and A2 to synapses in T2, the graph
        will have nodes A1 = 'A_T1' and A2 = 'A_T2'.
        """
        names = get_nodes_data.load_data_neuron_set(  # retrieve data
            ids=list(self.uid["body_id"].values),
            attributes=["systematicType:string"],
        )
        self.uid = pd.merge(
            self.uid,
            names,
            left_on="body_id",
            right_on=":ID(Body-ID)",
            how="left",
        )
        self.uid = self.uid.drop(columns=":ID(Body-ID)")
        self.uid = self.uid.rename(columns={"systematicType:string": "node_label"})

        if split_neurons is None:
            return
        for neuron in split_neurons:
            subdivisions = neuron.get_subdivisions()
            assert subdivisions is not None, "Neuron subdivisions is None"
            unique_starts = subdivisions[
                ["start_id", "subdivision_start", "subdivision_start_name"]
            ].drop_duplicates()
            for _, row in unique_starts.iterrows():
                id_tuple = (row["start_id"], row["subdivision_start"])
                self.uid.loc[self.uid["neuron_ids"] == id_tuple, "node_label"] += row[
                    "subdivision_start_name"
                ]

    def __build_adjacency_matrices(
        self, nodelist: Optional[list[UID] | list[int]] = None
    ):
        """
        Create a dictionary with relevant connectivity matrices
        and the index ordering table.
        """
        # get only the nodes that have a connection n the graph
        if nodelist is None:
            nodelist = self.uid["uid"].to_list()

        nodelist = [n for n in nodelist if n in self.graph.nodes]

        mat_norm = nx.to_scipy_sparse_array(
            self.graph, nodelist=nodelist, weight="eff_weight_norm", format="csr"
        )
        mat_unnorm = nx.to_scipy_sparse_array(
            self.graph, nodelist=nodelist, weight="eff_weight", format="csr"
        )
        mat_syn_count = nx.to_scipy_sparse_array(
            self.graph, nodelist=nodelist, weight="syn_count", format="csr"
        )
        lookup = pd.DataFrame(
            data={
                "index": np.arange(len(nodelist)),
                "uid": nodelist,
                "body_id": self.__convert_uid_to_neuron_ids(
                    nodelist, output_type="body_id"
                ),
                "subdivision": self.__convert_uid_to_neuron_ids(
                    nodelist, output_type="subdivision"
                ),
            }
        )
        lookup = lookup.sort_values("index")

        self.adjacency = {
            "mat_norm": mat_norm,
            "mat_unnorm": mat_unnorm,
            "mat_syncount": mat_syn_count,
            "lookup": lookup,
        }
        return

    def __get_node_attributes(
        self,
        attribute: typing.Literal["uid", "body_id"] | NeuronAttribute,
        default="None",
    ):
        """
        Get the node attributes for the graph.
        If the attribute is not present, identify it in the nodes dataframe and add it to the graph.
        The attributes are then defined based on the initial neuron from which it is defined.
        For instance if a neuron A is subdivided in A1 and A2, the neurotransmitter type
        of (A,1) and (A,2) will be the same as A.
        """
        if attribute not in self.__defined_attributes_in_graph:
            if (
                attribute == "uid"
            ):  # should be there from default, but present for backward compatibility
                attr_list = {node: node for node in self.graph.nodes}
                nx.set_node_attributes(self.graph, attr_list, attribute)
                self.__defined_attributes_in_graph.append(attribute)
                return self.graph.nodes
            elif attribute == "body_id":
                attr_list = {
                    node: self.uid.loc[self.uid["uid"] == node, "body_id"].values[0]
                    for node in self.graph.nodes
                }  # type: ignore
                nx.set_node_attributes(self.graph, attr_list, attribute)
                self.__defined_attributes_in_graph.append(attribute)
                return nx.get_node_attributes(self.graph, attribute)
            else:
                print(f"Attribute {attribute} not found in the graph. Adding it.")
                nodes = self.get_nodes(type="uid")  # uids
                body_ids = self.get_nodes(type="body_id")
                attr = get_nodes_data.load_data_neuron_set(  # retrieve data
                    ids=body_ids,
                    attributes=[attribute],
                )
                attr_mapping = {
                    body_id: value  # map body id to attribute value
                    for body_id, value in zip(attr[":ID(Body-ID)"], attr[attribute])
                }
                attr_list = {
                    node: attr_mapping[body_id]  # map node to attribute value
                    for node, body_id in zip(nodes, body_ids)
                }

                # replace None values with a default value
                attr_list = {
                    k: default if v is None else v for k, v in attr_list.items()
                }
                nx.set_node_attributes(self.graph, attr_list, attribute)
                self.__defined_attributes_in_graph.append(attribute)
            return nx.get_node_attributes(self.graph, attribute)

    # public methods
    # --- copy
    def subgraph(
        self,
        nodes: Optional[typing.Iterable[UID] | typing.Iterable[int]] = None,
        edges: Optional[list[tuple[UID, UID]] | list[tuple[int, int]]] = None,
    ):
        """
        Copy operator of the Connections class that returns a new object
        generated from the subgraph of the current object.
        All existing connections between the set nodes are included.

        Parameters
        ----------
        nodes: list[int]
            List of nodes to consider in the subgraph. These are uids.
            If None, the entire graph is considered.
        edges: list[tuple[int]]
            List of edges to consider in the subgraph.
            If None, all existing connections between the nodes are considered.
            nb: if both nodes and edges exist, the intersection of the two is
            considered.

        Returns
        -------
        Connections
            New Connections object with the subgraph.
        """
        # copy the original object
        new_connection_obj = copy.deepcopy(self)

        if nodes is None and edges is None:
            return new_connection_obj 

        # recompute all the attributes inside from the subset
        new_connection_obj.empty()

        # Identify the connections to keep
        connections_ = self.get_dataframe()
        if nodes is not None:
            connections_ = connections_[
                connections_["start_uid"].isin(nodes)
                & connections_["end_uid"].isin(nodes)
            ]
        else:
            # edges is necessarily not None. otherwise already returned above
            # get all the elements present in the tuples edges
            nodes = list(set([x for y in edges for x in y]))
        if edges is not None:
            rows_to_drop = []
            for index, row in connections_.iterrows():
                if (row["start_uid"], row["end_uid"]) not in edges:
                    rows_to_drop.append(index)
            connections_ = connections_.drop(rows_to_drop)

        # Update the neuron list fields
        new_connection_obj.neurons_pre_ = connections_[
            ":START_ID(Body-ID)"
            ].to_list()
        new_connection_obj.neurons_post_ = connections_[
            ":END_ID(Body-ID)"
            ].to_list()
        

        # update the effective weights
        connections_ = connections_.drop(
            columns=[
                "syn_count_norm",
                "eff_weight_norm",
            ]
        )
        new_connection_obj.set_connections(connections_)
        new_connection_obj.__compute_effective_weights_in_connections(ids="uid")

        # Subset the uids, keeps the names
        new_connection_obj.uid = copy.deepcopy(self.uid.loc[self.uid["uid"].isin(nodes)])
        # Initialize the graph
        new_connection_obj.set_graph()
        # Initialize the adjacency matrices
        new_connection_obj.set_adjacency_matrices()
        return new_connection_obj

    def subgraph_from_paths(
            self,
            source: UID | int | list[UID] | list[int],
            target: UID | int | list[UID] | list[int],
            n_hops: int,
            keep_edges: typing.Literal[
                'direct', 'intermediate', 'all'
                ] = 'intermediate',
        ):
        """
        Get the subgraph as new Connections object between a source node and a
        list of target nodes, up to n hops.

        Parameters
        ----------
        source: int or list[int]
            Source node(s) to consider.
        target: int or list[int]
            Target node(s) to consider.
        n_hops: int
            Number of hops to consider.
        keep_edges: str
            Type of edges to keep in the subgraph.
            Can be 'direct' (only edges directly involved in length n paths),
            'intermediate' (all edges between nodes in length n path except for
            edges between different target nodes), or 'all'.

        Returns
        -------
        Connections
            New Connections object with the subgraph.
        """
        graph_ = self.paths_length_n(n_hops, source, target)

        match keep_edges:
            case 'direct':
                return self.subgraph(
                    nodes = graph_.nodes(),
                    edges = graph_.edges()
                    )
            case 'intermediate':
                # get edges that are not between target nodes
                edges_to_keep = [
                    e for e in graph_.edges()
                    if not (e[0] in target and e[1] in target)
                ]
                return self.subgraph(
                    nodes = graph_.nodes(),
                    edges = edges_to_keep
                    )
            case 'all':
                return self.subgraph(
                    nodes = graph_.nodes(),
                    )
            case _:
                raise ValueError(
                    f"Class Connections::: > \
                    subgraph_from_paths(): Unknown keep_edges type {keep_edges}"
                )
                return 

    # --- getters
    def get_connections_with_only_traced_neurons(self):
        """
        Remove the neurons in the graph for which the field "status:string"
        is not "Traced".

        Returns
        -------
        Connections
            New Connections object with only the traced neurons.
        """
        neuron_tracing_status = self.get_node_attribute(
            self.get_nodes(), "status:string"
        )
        traced_nodes = [
            node
            for node, status in zip(self.get_nodes(), neuron_tracing_status)
            if status == "Traced"
        ]
        return self.subgraph(traced_nodes)

    def get_neuron_bodyids(self, selection_dict: Optional[SelectionDict] = None):
        """
        Get the neuron Body-IDs from the nodes dataframe based on a selection dictionary.
        """
        nodes = self.get_nodes(type="body_id")
        return get_nodes_data.get_neuron_bodyids(selection_dict, nodes)

    def get_neuron_ids(
        self,
        selection_dict: Optional[SelectionDict] = None,
    ) -> list[UID]:
        """
        Get the neuron IDs from the nodes dataframe as loaded in the initial
        dataset, based on a selection dictionary.
        """
        nodes = self.get_nodes(type="body_id")
        body_ids = get_nodes_data.get_neuron_bodyids(selection_dict, nodes)
        return self.__get_uids_from_bodyids(body_ids)

    def get_neurons_pre(self):
        return self.neurons_pre

    def get_neurons_post(self):
        return self.neurons_post

    def get_dataframe(
        self,
        specific_start_uids: Optional[list[int]] = None,
        specific_end_uids: Optional[list[int]] = None,
    ):
        """
        Get the connections table.
        """
        if specific_start_uids is not None and specific_end_uids is not None:
            return self.connections[
                self.connections["start_uid"].isin(specific_start_uids)
                & self.connections["end_uid"].isin(specific_end_uids)
            ]
        if specific_start_uids is not None:
            return self.connections[
                self.connections["start_uid"].isin(specific_start_uids)
            ]
        if specific_end_uids is not None:
            return self.connections[self.connections["end_uid"].isin(specific_end_uids)]
        return self.connections

    def get_graph(
        self,
        weight_type: typing.Literal[
            "syn_count", "eff_weight", "syn_count_norm", "eff_weight_norm"
        ] = "eff_weight",
        syn_threshold: Optional[int] = None,
    ):
        """
        Get the graph of the connections.
        """
        if weight_type not in [
            "syn_count",
            "eff_weight",
            "syn_count_norm",
            "eff_weight_norm",
        ]:
            raise ValueError(
                f"Class Connections \
                ::: > get_graph(): Unknown weight type {weight_type}"
            )
        graph_ = self.graph.copy()
        assert isinstance(graph_, nx.DiGraph)
        nx.set_edge_attributes(
            graph_,
            {(u, v): {"weight": d[weight_type]} for u, v, d in graph_.edges(data=True)},
        )
        if syn_threshold is not None:
            edges_to_remove = [
                (u, v)
                for u, v, d in graph_.edges(data=True)
                if abs(d["weight"]) < syn_threshold
            ]
            graph_.remove_edges_from(edges_to_remove)
        return graph_

    def get_nt_weights(self):
        return self.nt_weights

    def get_adjacency_matrix(
        self,
        type_: typing.Literal["norm", "unnorm", "syn_count"] = "syn_count",
        hops: int = 1,
        intermediates: bool = True,
    ):
        assert (
            self.adjacency is not None
        ), "Error: adjacency matrix is None - probably wasn't initialised"
        match type_:
            case "norm":
                mat_ = self.adjacency["mat_norm"]
            case "unnorm":
                mat_ = self.adjacency["mat_unnorm"]
            case "syn_count":
                mat_ = self.adjacency["mat_syncount"]
            case _:
                raise ValueError(
                    f"Class Connections::: > get_adjacency_matrix(): Unknown type {type_}"
                )
        if intermediates:
            return matrix_utils.connections_up_to_n_hops(mat_, hops)
        return matrix_utils.connections_at_n_hops(mat_, hops)

    def get_cmatrix(
        self, type_: typing.Literal["norm", "unnorm", "syn_count"] = "norm"
    ):  # cmatrix object
        """
        Returns the cmatrix object of the connections.
        By default, the normalized adjacency matrix is used.
        """
        return cmatrix.CMatrix(
            self.get_adjacency_matrix(type_=type_), self.get_lookup()
        )

    def get_lookup(self):
        assert (
            self.adjacency is not None
        ), "Error: adjacency matrix is None - probably wasn't initialised"
        return self.adjacency["lookup"]

    def get_nx_graph(
        self,
        hops: int = 1,
        nodes: Optional[list[int]] = None,
    ):
        """
        Get the effective graph up to n hops.

        Parameters
        ----------
        hops: int
            Number of hops to consider.
        nodes: list[int]
            List of nodes to consider in the subgraph.
            If None, the entire graph is considered.

        Returns
        -------
        nx.DiGraph
            Graph with the effective connections up to n hops.
        Modifies
        -------
            adds the graph to the subgraphs dictionary.
        """
        if hops == 1:
            graph_ = self.graph
        else:
            if f"graph_{hops}_hops" not in self.subgraphs.keys():
                self.__compute_n_hops(hops, initialize_graph=True)
            graph_ = self.subgraphs[f"graph_{hops}_hops"]
        if nodes is not None:
            graph_ = nx_utils.get_subgraph(graph_, nodes)
        return graph_

    @typing.overload
    def get_nodes(self, type: typing.Literal["uid"] = "uid") -> list[UID]: ...
    @typing.overload
    def get_nodes(
        self, type: typing.Literal["body_id"] = "body_id"
    ) -> list[BodyId]: ...

    def get_nodes(
        self, type: typing.Literal["uid", "body_id"] = "uid"
    ) -> list[UID] | list[BodyId]:
        """
        Get the nodes of the graph.
        """
        if type == "uid":
            return list(self.graph.nodes)
        elif type == "body_id":
            return list(nx.get_node_attributes(self.graph, "body_id").values())
        else:
            raise ValueError(
                f"Class Connections \
                ::: > get_nodes(): Unknown type {type}"
            )

    def get_nb_synapses(
        self,
        start_id: int,
        end_id: int,
        input_type: typing.Literal["uid", "body_id"] = "uid",
    ):
        """
        Get the number of synapses between two neurons.
        """
        if input_type == "body_id":
            table = self.connections[
                (self.connections[":START_ID(Body-ID)"] == start_id)
                & (self.connections[":END_ID(Body-ID)"] == end_id)
            ]
            # drop subdivision duplicates in the post synaptic neuron
            # (from duplication policy in neuron splitting)
            table = table.drop_duplicates(
                subset=[":START_ID(Body-ID)", "subdivision_start", ":END_ID(Body-ID)"]
            )
            nb_synapses = table["syn_count"].sum()
        elif input_type == "uid":
            table = self.connections[
                (self.connections["start_uid"] == start_id)
                & (self.connections["end_uid"] == end_id)
            ]
            nb_synapses = table["syn_count"].sum()

        else:
            raise ValueError(
                f"Class Connections \
                ::: > get_nb_synapses(): Unknown input type {input_type}"
            )
        return nb_synapses

    def subset_uids_on_label(self, label_contains: str):
        """
        Get the uids of the neurons that have a label containing the given string.
        """
        return self.uid.loc[
            self.uid["node_label"].str.contains(label_contains, na=False)
        ]["uid"].values

    def get_node_label(self, uid: UID | int):
        """
        Get the label of a node.
        """
        return self.uid.loc[self.uid["uid"] == uid]["node_label"].values[0]

    def get_node_attribute(
        self, uid: list[UID] | list[int] | UID | int, attribute: NeuronAttribute
    ) -> list:
        """
        Get the attribute of a node.

        uid: int or list[int]
        """
        self.__get_node_attributes(attribute)  # defines it if not present
        all = nx.get_node_attributes(self.graph, attribute)
        if isinstance(uid, list):
            return [all[u] for u in uid]
        else:
            return all[uid]

    def get_neurons_in_neuropil(self, neuropil: str, side: Optional[str] = None):
        """
        Get the uids of neurons in a given neuropil.
        If a side is given, only the neurons on that side are considered.
        """
        if side is not None:
            return self.get_neuron_ids(
                {
                    "somaSide:string": side,
                    "somaNeuromere:string": neuropil,
                }
            )
        else:
            return self.get_neuron_ids(
                {
                    "somaNeuromere:string": neuropil,
                }
            )

    @typing.overload
    def get_neurons_downstream_of(
        self,
        neuron_id: int,
        input_type: typing.Literal["uid", "body_id"] = "uid",
        output_type: typing.Literal["uid"] = "uid",
    ) -> list[UID]: ...
    @typing.overload
    def get_neurons_downstream_of(
        self,
        neuron_id: int,
        input_type: typing.Literal["uid", "body_id"] = "uid",
        output_type: typing.Literal["body_id"] = "body_id",
    ) -> list[BodyId]: ...

    def get_neurons_downstream_of(
        self,
        neuron_id: int,
        input_type: typing.Literal["uid", "body_id"] = "uid",
        output_type: typing.Literal["uid", "body_id"] = "uid",
    ) -> list[UID] | list[BodyId]:
        """
        Get the neurons downstream of a given neuron, based on the graph.

        Parameters
        ----------
        neuron_id: int
            Identifier of the neuron.
        input_type: str
            Type of the input list, can be 'uid' or 'body_id'.
        return_type: str
            Type of the return list, can be 'uid' or 'body_id'.

        Returns
        -------
        list[int]
            List of identifiers of the downstream neurons.
        """
        # Get uid of the neuron as the graph indexes with uids
        if input_type == "uid":
            nid = neuron_id
        elif input_type == "body_id":
            nid = self.get_first_uid_from_bodyid(neuron_id)
        else:
            raise ValueError(
                f"Class Connections \
                ::: > get_neurons_downstream_of(): Unknown input type {input_type}"
            )

        # Get the downstream neurons
        downstream = [node for node in self.graph.successors(nid)]

        # Convert the output type
        if output_type == "uid":
            return list(downstream)
        elif output_type == "body_id":
            return self.__convert_uid_to_neuron_ids(downstream, output_type="body_id")
        else:
            raise ValueError(
                f"Class Connections \
                ::: > get_neurons_downstream_of(): Unknown output type {output_type}"
            )

    @typing.overload
    def get_neurons_upstream_of(
        self,
        neuron_id: int,
        input_type: typing.Literal["uid", "body_id"] = "uid",
        output_type: typing.Literal["uid"] = "uid",
    ) -> list[UID]: ...
    @typing.overload
    def get_neurons_upstream_of(
        self,
        neuron_id: int,
        input_type: typing.Literal["uid", "body_id"] = "uid",
        output_type: typing.Literal["body_id"] = "body_id",
    ) -> list[BodyId]: ...

    def get_neurons_upstream_of(
        self,
        neuron_id: int,
        input_type: typing.Literal["uid", "body_id"] = "uid",
        output_type: typing.Literal["uid", "body_id"] = "uid",
    ) -> list[UID] | list[BodyId]:
        """
        Get the neurons upstream of a given neuron, based on the graph.

        Parameters
        ----------
        neuron_id: int
            Identifier of the neuron.
        input_type: str
            Type of the input list, can be 'uid' or 'body_id'.
        return_type: str
            Type of the return list, can be 'uid' or 'body_id'.

        Returns
        -------
        list[int]
            List of identifiers of the upstream neurons.
        """
        # Get uid of the neuron as the graph indexes with uids
        if input_type == "uid":
            nid = neuron_id
        elif input_type == "body_id":
            nid = self.get_first_uid_from_bodyid(neuron_id)
        else:
            raise ValueError(
                f"Class Connections \
                ::: > get_neurons_downstream_of(): Unknown input type {input_type}"
            )

        # Get the upstream neurons
        upstream = [node for node in self.graph.predecessors(nid)]

        # Convert the output type
        if output_type == "uid":
            return list(upstream)
        elif output_type == "body_id":
            return self.__convert_uid_to_neuron_ids(upstream, output_type="body_id")
        else:
            raise ValueError(
                f"Class Connections ::: > get_neurons_upstream_of():\
                Unknown output type {output_type}"
            )

    def get_bodyids_from_uids(
        self, uids: UID | list[UID] | list[int] | set[UID] | set[int]
    ) -> list[BodyId]:
        """
        Get the body ids from the uids.
        """
        if isinstance(uids, int):
            uids = [uids]
        if isinstance(uids, set):
            uids = list(uids)
        return self.__convert_uid_to_neuron_ids(uids, output_type="body_id")

    def get_first_uid_from_bodyid(self, body_id: get_nodes_data.BodyId | int) -> UID:
        """
        Get the first uid (if there are more than one) corresponding to a body id.
        """
        return self.__get_uids_from_bodyids([body_id])[0]

    def get_uids_from_bodyid(self, body_id: get_nodes_data.BodyId | int) -> list[UID]:
        """
        Get the uids corresponding to a body id.
        """
        return self.__get_uids_from_bodyids([body_id])

    def get_uids_from_bodyids(self, body_ids: list[BodyId] | list[int]) -> list[UID]:
        """
        Get all the uids from a list of body ids.
        """
        return self.__get_uids_from_bodyids(body_ids)

    def get_out_degree(self, uid: UID | int):
        """
        Get the out degree of a node.
        """
        return self.graph.out_degree(uid)

    # --- setters
    def merge_nodes(self, nodes: list[UID] | list[int]):
        """
        Merge a list of nodes into a single node.
        The first node in the list is kept as reference.
        """
        ref_node, nodes = nodes[0], nodes[1:]
        for other_node in nodes:
            # merge the nodes in the graph
            self.graph = nx.contracted_nodes(
                self.graph,
                ref_node,
                other_node,
            )
            # merge the nodes in the connections dataframe
            self.connections = self.connections.replace(
                to_replace=other_node,
                value=ref_node,
            )
        # clear the connections table from duplicates:
        # if a connection is present in both nodes, the weight is summed
        self.connections = (
            self.connections.groupby(["start_uid", "end_uid", "predictedNt:string"])
            .agg(
                syn_count=("syn_count", "sum"),
                eff_weight=("eff_weight", "sum"),
                syn_count_norm=("syn_count_norm", "sum"),
                eff_weight_norm=("eff_weight_norm", "sum"),
            )
            .reset_index()
        )

        # update the adjacency matrices
        self.adjacency = self.__build_adjacency_matrices()

        return

    def reorder_neurons(
        self,
        by: typing.Literal["list"] = "list",
        order: Optional[list[UID] | list[int]] = None,
        order_type: typing.Literal["index", "uid"] = "uid",
    ):
        """
        Reorder the neurons in the adjacency matrices and the corresponding lookup table.

        Parameters
        ----------
        by: str
            Method to reorder the neurons.
            Can be 'list' to reorder by a given list of neuron IDs.
        order: list[int]
            List of neuron IDs to reorder the adjacency matrices by.
            Warning: the list type is given by the order_type parameter.
        order_type: str
            Type of the order list.
            Can be 'uid' for neuron IDs or 'index' for the adjacency matrix index.
        """
        match by:
            case "list":
                if order_type == "uid":
                    order_ = order
                elif order_type == "index":
                    assert (
                        self.adjacency is not None
                    ), "Error: adjacency matrix is None - probably wasn't initialised"
                    order_ = self.adjacency["lookup"].loc[order, "uid"].to_list()
                else:
                    raise ValueError(
                        f"Class Connections > reorder_neurons(): \
                            Unknown order type {order_type}"
                    )
                self.adjacency = self.__build_adjacency_matrices(nodelist=order_)
            case _:
                raise ValueError(
                    f"Class Connections::: > reorder_neurons(): Unknown method {by}"
                )
        return

    def set_connections(self, connections: pd.DataFrame):
        self.connections = connections
        return

    def set_graph(self, graph: Optional[nx.DiGraph] = None):
        if graph is None:
            self.__build_graph()
        else:
            self.graph = graph

    def set_nt_weights(self, nt_weights: dict[str, int]):
        self.nt_weights = nt_weights
        return

    def set_adjacency_matrices(self):
        self.__build_adjacency_matrices()
        return

    def include_node_attributes(self, attributes: list[NeuronAttribute]):
        """
        Include node attributes in the graph.
        """
        for attribute in attributes:
            _ = self.__get_node_attributes(attribute)
        return

    # --- computations
    def __compute_n_hops(self, n: int, initialize_graph: bool = False):
        """
        Compute the n-hop adjacency matrix and matching effective connections graph.
        """
        # TODO: edit such that it creates a new Connections object instead
        # compute the n-hop adjacency matrix
        assert (
            self.adjacency is not None
        ), "Error: adjacency matrix is None - probably wasn't initialised"
        self.adjacency[f"mat_norm_{n}_hops"] = matrix_utils.connections_up_to_n_hops(
            self.adjacency["mat_norm"], n
        )
        # create the associated graph
        if initialize_graph:
            self.subgraphs[f"graph_{n}_hops"] = nx.from_scipy_sparse_array(
                self.adjacency[f"mat_norm_{n}_hops"],
                create_using=nx.DiGraph,
            )
            nodes_data_ = dict(self.graph.nodes(data=True))
            nx.set_node_attributes(
                self.subgraphs[f"graph_{n}_hops"],
                nodes_data_,
            )
        return

    def paths_length_n(
        self,
        n: int,
        source: UID | int | list[UID] | list[int],
        target: UID | int | list[UID] | list[int],
        syn_threshold: Optional[int] = None,
    ):
        """
        Get the graph with all the paths of length n between two sets of nodes.

        Parameters
        ----------
        n: int
            Length of the paths.
        source: list[int]
            List of source nodes.
        target: list[int]
            List of target nodes.
        syn_threshold: int
            Threshold for the synapse count.

        Returns
        -------
        nx.DiGraph
            Graph composed of all the nodes reached by the paths of length n
            between the source and target nodes.
        """
        if isinstance(source, int):
            source = [source]
        if isinstance(target, int):
            target = [target]
        # Find all the paths of length n between the source and target nodes
        edges_ = []
        for s in source:
            target_ = list(set(target) - {s})  # remove the source from the target
            subedges_ = nx.all_simple_edge_paths(
                self.graph, source=s, target=target_, cutoff=n
            )
            _ = [edges_.extend(path) for path in subedges_]
        # Filter the edges based on the synapse threshold
        if syn_threshold is not None:
            edges_ = [
                edge
                for edge in edges_
                if self.graph[edge[0]][edge[1]]["syn_count"] >= syn_threshold
            ]
        # Create the subgraph
        return nx_utils.get_subgraph_from_edges(self.graph, edges_)

    def cluster_hierarchical(
        self, reference: typing.Literal["norm", "unnorm", "syn_count"] = "syn_count"
    ):
        """
        Cluster the adjacency matrix hierarchically.

        Parameters
        ----------
        reference: str
            Attribute to use as reference for the clustering.
            Default is the synapse count.

        Returns
        -------
        scipy.sparse.csr_matrix
            Clustered adjacency matrix.

        Modifies
        -------
            Reorders the neurons in the adjacency matrices.
        """
        mat = self.get_adjacency_matrix(reference)
        ordered_mat_, order_ = matrix_utils.cluster_matrix_hierarchical(mat)
        self.reorder_neurons(by="list", order=order_, order_type="index")
        return ordered_mat_

    # --- display
    def display_adjacency_matrix(
        self,
        type_: typing.Literal["norm", "unnorm", "syn_count"] = "syn_count",
        method: typing.Literal["spy", "heatmap"] = "spy",
        title: str = "test",
    ):
        """
        Display the adjacency matrix.
        """
        plt.figure(figsize=FIGSIZE, dpi=DPI)
        match method:
            case "spy":
                plt.spy(self.get_adjacency_matrix(type_), markersize=0.1)
            case "heatmap":
                mat = self.get_adjacency_matrix(type_)
                sns.heatmap(
                    mat.toarray(),
                    cmap=params.red_heatmap,
                    mask=(mat.toarray() == 0),
                    # xticklabels=False,
                    # yticklabels=False,
                )
            case _:
                raise ValueError(
                    f"Class Connections::: > display_adjacency_matrix(): Unknown method {method}"
                )

        plt.savefig(os.path.join(params.PLOT_DIR, title + "_matrix.pdf"))
        return

    @typing.overload
    def display_graph(
        self,
        pos: Optional[typing.Mapping] = None,
        method: typing.Literal[
            "circular", "spring", "kamada_kawai", "breadth_first", "force", "spectral"
        ] = "circular",
        title: str = "test",
        label_nodes: bool = False,
        syn_threshold: Optional[int] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        save: bool = True,
        return_pos: typing.Literal[False] = False,
    ) -> matplotlib.axes.Axes: ...
    @typing.overload
    def display_graph(
        self,
        pos: Optional[typing.Mapping] = None,
        method: typing.Literal[
            "circular", "spring", "kamada_kawai", "breadth_first", "force", "spectral"
        ] = "circular",
        title: str = "test",
        label_nodes: bool = False,
        syn_threshold: Optional[int] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        save: bool = True,
        return_pos: typing.Literal[True] = True,
    ) -> tuple[matplotlib.axes.Axes, Mapping]: ...

    def display_graph(
        self,
        pos: Optional[typing.Mapping] = None,
        method: typing.Literal[
            "circular", "spring", "kamada_kawai", "breadth_first", "force", "spectral"
        ] = "circular",
        title: str = "test",
        label_nodes: bool = False,
        syn_threshold: Optional[int] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        save: bool = True,
        return_pos: bool = False,
    ):
        """
        Display the graph.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        assert ax is not None  # needed for type hinting
        # restrict the number of edges visualised to the threshold weight
        graph_ = nx_utils.threshold_graph(self.graph, syn_threshold)

        if pos is None:
            match method:
                case "circular":
                    pos = nx.circular_layout(graph_)
                case "spring":
                    pos = nx.arf_layout(graph_)
                case "kamada_kawai":
                    graph_ = self.get_graph(
                        weight_type="syn_count",
                        syn_threshold=syn_threshold,
                    )  # no negative weights
                    pos = nx.kamada_kawai_layout(graph_)
                case "breadth_first":
                    pos = nx.bfs_layout(graph_)
                case "force":
                    pos = nx.forceatlas2_layout(graph_)
                case "spectral":
                    pos = nx.spectral_layout(graph_)
                case _:
                    raise ValueError(
                        f"Class Connections > display_graph(): Unknown method {method}"
                    )
        ax = nx_design.draw_graph(
            graph_, pos, ax=ax, return_pos=False, label_nodes=label_nodes
        )
        ax.set_title(title)
        if save:
            plt.savefig(os.path.join(params.PLOT_DIR, title + "_graph.pdf"))
        if return_pos:
            return ax, pos
        return ax

    def display_graph_per_attribute(
        self,
        attribute: NeuronAttribute = "exitNerve:string",
        center: Optional[list[UID] | list[int] | str] = None,
        syn_threshold: Optional[int] = None,
        title: str = "test",
        save: bool = True,
    ):
        """
        Display the graph per node attribute grouping (default neuropil).
        Neurons are aggregated by attribute in a circle layout.
        The centers of the attribute circles form a larger circle in turn.
        One specific neuron or attribute can be placed at the center.

        Parameters
        ----------
        attribute: str
            Node attribute to group by.
            Can be any attribute present in the nodes dataframe.
        center: list[int] or str
            Node IDs to place at the center.
            or attribute to place at the center.
        title: str
            Title of the plot.

        Returns
        -------
        None
        """
        # ensures the attribute is present in the graph
        _ = self.__get_node_attributes(attribute)
        # restrict the number of edges visualised to the threshold weight
        graph_ = nx_utils.threshold_graph(self.graph, syn_threshold)

        #  use the nx library to plot the graph grouped by the attribute
        if center is not None:
            if isinstance(center, str):
                ax = nx_design.draw_graph_grouped_by_attribute(
                    graph_,
                    attribute,
                    center_instance=center,
                )
            elif isinstance(center, int):
                center = [center]
                ax = nx_design.draw_graph_grouped_by_attribute(
                    graph_,
                    attribute,
                    center_nodes=center,
                )
            elif isinstance(center, list):
                ax = nx_design.draw_graph_grouped_by_attribute(
                    graph_,
                    attribute,
                    center_nodes=center,
                )
            else:
                center = list(center)
                ax = nx_design.draw_graph_grouped_by_attribute(
                    graph_,
                    attribute,
                    center_nodes=center,
                )
        else:
            ax = nx_design.draw_graph_grouped_by_attribute(
                graph_,
                attribute,
            )
        if save:
            plt.savefig(os.path.join(params.PLOT_DIR, title + "_sorted_graph.pdf"))
            return ax

    def plot_xyz(
        self,
        x: list[int],
        y: list[int],
        sorting: Optional[SortingStyle | NeuronAttribute] = None,
        title: str = "test",
    ):
        """
        Draw a 3D plot of the connections between two sets of neurons.

        Parameters
        ----------
        x: list[int]
            List of source nodes.
        y: list[int]
            List of target nodes.
        sorting: str or dict of str
            Attribute to sort the connections by.
            Can be 'degree', 'betweenness' for networkx attributes.
            Can by any attribute present in the nodes dataframe.
            Can be 'from_index' to sort by the order of the nodes in associated
            to the adjacency matrix.
        title: str
            Title of the plot.

        Returns
        -------
        None
        """
        graph_ = self.get_graph()
        assert isinstance(graph_, nx.DiGraph)
        # ensure that the sorting parameter is in the nodes of the graph
        if sorting in typing.get_args(NeuronAttribute):
            _ = self.__get_node_attributes(sorting)

        if sorting == "from_index":
            z = [node for node in self.graph.nodes if not node in x and not node in y]
            assert self.adjacency is not None  # needed for type hinting
            sort_list_nodes = self.adjacency["lookup"]["body_id"].to_list()
            # reorder x,y,z by the order of the adjacency matrix indexings
            x_ = [n for n in sort_list_nodes if n in x]
            y_ = [n for n in sort_list_nodes if n in y]
            z_ = [n for n in sort_list_nodes if n in z]
            pos = nx_design.position_3d_nodes(x_, y_, z_)
            # draw the plot
            _ = nx_design.plot_xyz(graph_, x, y, z=z, sorting=sorting, pos=pos)
        else:
            # restrict x and y to the nodes in the graph
            x = [n for n in x if n in graph_.nodes]
            y = [n for n in y if n in graph_.nodes]
            # draw the plot
            _ = nx_design.plot_xyz(graph_, x, y, sorting=sorting)
        # save the plot
        plt.savefig(os.path.join(params.PLOT_DIR, title + "3d_plot.pdf"))
        return

    def draw_3d_custom_axis(
        self,
        x: list[UID] | list[int],
        y: list[UID] | list[int],
        x_sorting: SortingStyle | NeuronAttribute | None = "degree",
        y_sorting: SortingStyle | NeuronAttribute | None = "exitNerve:string",
        z_sorting: SortingStyle | NeuronAttribute | None = "input_clustering",
        title: str = "test",
    ):
        """
        Draw a 3D plot of the connections between two sets of neurons.
        Each axis gets its own sorting parameter.
        """
        # ensure that the sorting parameters are in the nodes of the graph
        for sorting_method in [x_sorting, y_sorting, z_sorting]:
            if sorting_method not in [
                None,
                "degree",
                "betweenness",
                "from_index",
                "input_clustering",
                "centrality",
                "output_clustering",
            ]:
                _ = self.__get_node_attributes(sorting_method)
        # restrict x and y to the nodes in the graph
        graph_ = self.get_graph()
        x = [n for n in x if n in graph_.nodes]
        y = [n for n in y if n in graph_.nodes]
        # draw the plot
        _ = nx_design.plot_xyz(
            graph_,
            x,
            y,
            sorting=[x_sorting, y_sorting, z_sorting],
        )
        plt.savefig(os.path.join(params.PLOT_DIR, title + "3dx_plot.pdf"))

    @typing.overload
    def draw_graph_concentric_by_attribute(
        self,
        center_nodes: list[UID] | list[int],
        target_nodes: list[UID] | list[int],
        attribute: NeuronAttribute | typing.Literal["uid", "body_id"],
        syn_threshold: Optional[int] = None,
        label_nodes: bool = False,
        title: str = "test",
        save: bool = True,
        ax: Optional[matplotlib.axes.Axes] = None,
        return_pos: typing.Literal[False] = False,
    ) -> matplotlib.axes.Axes: ...
    @typing.overload
    def draw_graph_concentric_by_attribute(
        self,
        center_nodes: list[UID] | list[int],
        target_nodes: list[UID] | list[int],
        attribute: NeuronAttribute | typing.Literal["uid", "body_id"],
        syn_threshold: Optional[int] = None,
        label_nodes: bool = False,
        title: str = "test",
        save: bool = True,
        ax: Optional[matplotlib.axes.Axes] = None,
        return_pos: typing.Literal[True] = True,
    ) -> tuple[matplotlib.axes.Axes, Mapping]: ...

    def draw_graph_concentric_by_attribute(
        self,
        center_nodes: list[UID] | list[int],
        target_nodes: list[UID] | list[int],
        attribute: NeuronAttribute | typing.Literal["uid", "body_id"],
        syn_threshold: Optional[int] = None,
        label_nodes: bool = False,
        title: str = "test",
        save: bool = True,
        ax: Optional[matplotlib.axes.Axes] = None,
        return_pos: bool = False,
    ) -> matplotlib.axes.Axes | tuple[matplotlib.axes.Axes, Mapping]:
        """
        Represent the graph with 3 concentric circles.
        center_nodes are on a center circle, typically input neurons.
        On the outer circle are target_nodes grouped by attribute.
        On the intermediate circle are the rest of the nodes in the graph,
        with their angle minimising distance to the nodes they're connected to
        for readability.
        """
        # ensures the attribute is present in the graph
        _ = self.__get_node_attributes(attribute)
        # threshold synapses for visualisation
        graph_ = nx_utils.threshold_graph(self.graph, syn_threshold)
        center_nodes = [n for n in center_nodes if n in graph_.nodes]
        target_nodes = [n for n in target_nodes if n in graph_.nodes]
        # draw the graph
        ax, pos = nx_design.draw_graph_concentric_by_attribute(
            graph=graph_,
            center_nodes=center_nodes,
            target_nodes=target_nodes,
            attribute=attribute,
            ax=ax,
            return_pos=True,
            label_nodes=label_nodes,
        )
        ax.set_title(title)
        if save:
            plt.savefig(os.path.join(params.PLOT_DIR, title + "_sorted_graph.pdf"))
        if return_pos:
            return ax, pos
        return ax

    @typing.overload
    def draw_graph_in_out_center_circle(
        self,
        input_nodes: list[UID] | list[int],
        output_nodes: list[UID] | list[int],
        syn_threshold: Optional[int] = None,
        label_nodes: bool = False,
        title: str = "test",
        save: bool = True,
        ax: Optional[matplotlib.axes.Axes] = None,
        return_pos: typing.Literal[False] = False,
    ) -> matplotlib.axes.Axes: ...
    @typing.overload
    def draw_graph_in_out_center_circle(
        self,
        input_nodes: list[UID] | list[int],
        output_nodes: list[UID] | list[int],
        syn_threshold: Optional[int] = None,
        label_nodes: bool = False,
        title: str = "test",
        save: bool = True,
        ax: Optional[matplotlib.axes.Axes] = None,
        return_pos: typing.Literal[True] = True,
    ) -> tuple[matplotlib.axes.Axes, Mapping]: ...

    def draw_graph_in_out_center_circle(
        self,
        input_nodes: list[UID] | list[int],
        output_nodes: list[UID] | list[int],
        syn_threshold: Optional[int] = None,
        label_nodes: bool = False,
        title: str = "test",
        save: bool = True,
        ax: Optional[matplotlib.axes.Axes] = None,
        return_pos: bool = False,
    ) -> matplotlib.axes.Axes | tuple[matplotlib.axes.Axes, Mapping]:
        """
        Draw the graph with the input neurons at the top,
        the output neurons at the bottom and the rest in the center placed on
        a circle.
        """
        # threshold synapses for visualisation
        graph_ = nx_utils.threshold_graph(self.graph, syn_threshold)
        input_nodes = [n for n in input_nodes if n in graph_.nodes]
        output_nodes = [n for n in output_nodes if n in graph_.nodes]
        # draw the graph
        ax, pos = nx_design.draw_graph_in_out_center_circle(
            graph=graph_,
            top_nodes=input_nodes,
            bottom_nodes=output_nodes,
            ax=ax,
            return_pos=True,
            label_nodes=label_nodes,
        )
        ax.set_title(title)
        if save:
            plt.savefig(os.path.join(params.PLOT_DIR, title + "_sorted_graph.pdf"))
        if return_pos:
            return ax, pos
        return ax

    def draw_bar_plot(
        self,
        neurons: typing.Iterable[UID] | typing.Iterable[int],
        attribute: NeuronAttribute = "class:string",
        ylabel: str = "# neurons",
        ax: Optional[matplotlib.axes.Axes] = None,
    ):
        """
        Draw a bar plot of the attribute of the neurons in the set.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        assert ax is not None  # needed for type hinting
        # Get the attribute values
        values = []
        for uid in neurons:
            values.append(self.get_node_attribute(uid, attribute))
        values.sort()
        values = pd.Series(values)
        counts = values.value_counts()
        counts.plot(kind="bar", ax=ax, colormap="grey")
        ax.set_xlabel(attribute)
        ax.set_ylabel(ylabel)
        ax = plots_design.make_nice_spines(ax)
        return ax

    # --- listing properties
    def list_possible_attributes(self):
        """
        List the attributes present in the nodes dataframe.
        """
        all_attributes = self.__defined_attributes_in_graph
        from_dataset = get_nodes_data.get_possible_columns()
        all_attributes.extend(from_dataset)
        all_attributes = np.unique(all_attributes)
        return all_attributes

    def list_neuron_properties(
        self,
        neurons: list[int | BodyId | UID],
        input_type: typing.Literal["uid", "body_id"] = "uid",
    ):
        """
        List the properties of the neurons in the list in a dataframe.
        Warning: only the properties already present in the graph are listed,
        which can vary depending on the processing done before.
        If you want include a specific property, use the
        Connections::include_node_attributes(attributes = [property]) method
        before calling this method.

        Parameters
        ----------
        neurons: list[int]
            List of neuron identifiers.
        input_type: str
            Type of the neuron identifiers, can be 'uid' or 'body_id'.

        Returns
        -------
        pd.DataFrame
            Dataframe with the properties of the neurons.
        """
        defined_properties = self.__defined_attributes_in_graph

        if input_type == "uid":
            nodes = list(neurons)
        elif input_type == "body_id":
            nodes = self.get_uids_from_bodyids(neurons)
        else:
            raise ValueError(
                f"Class Connections ::: > \
                list_neuron_properties(): Unknown neuron type {input_type}"
            )
        if len(nodes) < len(neurons):
            print("WARNING:: Some neurons were not found in the graph.")

        properties = pd.DataFrame(columns=defined_properties, index=nodes)

        for p in defined_properties:
            values = self.__get_node_attributes(p)
            properties[p] = [values[n] for n in nodes]

        properties["uid"] = nodes

        return properties

    # --- saving
    def save(self, name: str):
        """
        Save the connections object to a pickle file.
        """
        filename = os.path.join(params.CONNECTION_DIR, name + ".txt")
        with open(filename, "wb") as f:
            pickle.dump(self.__dict__, f)

    # --- clearing
    def empty(self):
        """
        Empty the connections object.
        """
        self.graph = None
        self.connections = None
        self.adjacency = None
        self.subgraphs = {}
        return