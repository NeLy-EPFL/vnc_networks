'''
Initializes the connections class.
Possible use cases are processing the adjacency matrix from a set of neurons,
or displaying the connections between neurons in a graph.
Useful class when juggeling between different representations of the connectome.

Each neuron is referenced by a tuple of (bodyId, subdivision) where the 
subdivision is a number indexing a set of synapses from the same neuron. This
allows to treat a single neuron as multiple nodes in the graph.
Each such created neuron has a unique identifier associated.

Use the following code to initialize the class:
```
from connections import Connections
neurons_pre = get_neurons_from_class('sensory neuron')
neurons_post = get_neurons_from_class('motor neuron')
connections = Connections(neurons_pre, neurons_post)
# connections.set_nt_weights({"acetylcholine": +1, "gaba": -1, "glutamate": -1, "unknown": 0, None: 0})
connections.initialize()
``` 
'''
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import seaborn as sns


import params
import get_nodes_data
import utils.nx_design as nx_design
import utils.matrix_utils as matrix_utils
import utils.nx_utils as nx_utils
import cmatrix
from neuron import Neuron

## ---- Constants ---- ##
FIGSIZE = (8, 8)
DPI = 300

## ---- Classes ---- ##

class Connections:
    def __init__(
            self,
            neurons_pre: list[int] = None,
            neurons_post: list[int] = None,
        ):
        '''
        If no neurons_post are given, the class assumes the same neurons as pre-synaptic.
        If no neurons_pre are given, the class considers all connections defined in the database.

        If not used as a standalone class, the initialize() method should be called after instantiation.
        '''
        if neurons_pre is None:
            neurons_ = pd.read_feather(params.NEUPRINT_NODES_FILE)
            neurons_pre_ = list(zip(
                neurons_[':ID(Body-ID)'].to_list(),
                np.zeros_like(neurons_[':ID(Body-ID)'].to_list())
            ))  # the subdivision is set to 0
        else:
            neurons_pre_ = list(zip(neurons_pre, np.zeros_like(neurons_pre)))
        self.neurons_pre = pd.MultiIndex.from_tuples(
            neurons_pre_,
            names=["body_id", "subdivision"],
            )
        if neurons_post is None:
            self.neurons_post = self.neurons_pre
        else:
            self.neurons_post = pd.MultiIndex.from_tuples(
                zip(list(
                    neurons_post,
                    np.zeros_like(neurons_post)
                    )),
                names=["body_id", "subdivision"],
                )
        self.nt_weights = params.NT_WEIGHTS
        self.subgraphs = {}
          

    # private methods
    def __get_connections(self):
        connections_ = pd.read_feather(params.NEUPRINT_CONNECTIONS_FILE)
        # filter out only the connections relevant here
        connections_ = connections_[
            connections_[":START_ID(Body-ID)"].isin(
                self.neurons_pre.get_level_values('body_id')
                )
            & connections_[":END_ID(Body-ID)"].isin(
                self.neurons_post.get_level_values('body_id')
                )
            ]
        
        # rename synapse count column explicitly and filter
        connections_["syn_count"] = connections_["weight:int"]
        connections_ = connections_[
            connections_["syn_count"] >= params.SYNAPSE_CUTOFF
            ]
        
        # add relevant information for the processing steps

        ## add the neurotransmitter type to the connections
        nttypes = get_nodes_data.load_data_neuron_set(
            self.neurons_pre.get_level_values('body_id'),
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
        ## Calculate effective weights normalized by total incoming synapses
        in_total = connections_.groupby(":END_ID(Body-ID)")["syn_count"].sum()
        in_total = in_total.to_frame(name="in_total")
        connections_ = connections_.merge(
            in_total, left_on=":END_ID(Body-ID)", right_index=True
            )
        connections_["syn_count_norm"] = connections_[
            "syn_count"
            ] / connections_["in_total"]
        connections_["eff_weight_norm"] = connections_[
            "eff_weight"
            ] / connections_["in_total"]
        connections_ = connections_.drop(columns=["in_total"])

        # add a column with 'subdivision_start' and 'subdivision_end' with zeros as values
        connections_["subdivision_start"] = 0
        connections_["subdivision_end"] = 0
        self.connections = connections_
        return

    def __split_neurons_in_connections(self, split_neurons: list[Neuron] = None):
        """
        Divide the neurons that have more than one subdivision into multiple nodes.
        Each subdivision is considered as a separate node, based on the list of
        synapses within.
        """
        pass
        # TODO: implement the split_neurons method
        # it should go in the Neurons, see which synapses are grouped, 
        # and split the neurons accordingly

    def __map_uid(self):
        """
        Map the tuples (bodyId, subdivision) to a unique identifier (uid) number that will 
        be used to reference the newly defined neurons, e.g. as graph node ids.
        """
        # reference all unique tuples in (":START_ID(Body-ID)", "subdivision_start")
        # and (":END_ID(Body-ID)", "subdivision_end") from the self.connections dataframe
        unique_objects = list(set(
            zip(self.connections[':START_ID(Body-ID)'], self.connections["subdivision_start"])
            ).union(set(
                zip(self.connections[":END_ID(Body-ID)"], self.connections[ "subdivision_end"])
            )))
        # create a new df mapping elements in the set to integers
        uids = list(range(len(unique_objects)))
        self.uid = pd.DataFrame({'uid':uids, 'neuron_ids':unique_objects})
        # add uids to the connections table
        self.connections['start_uid'] = self.__convert_neuron_ids_to_uid(
            self.connections[[':START_ID(Body-ID)','subdivision_start']].values,
            input_type = 'table',
        )
        self.connections['end_uid'] = self.__convert_neuron_ids_to_uid(
            self.connections[[':END_ID(Body-ID)','subdivision_end']].values,
            input_type = 'table',
        )
        return
    
    def __convert_neuron_ids_to_uid(
        self,
        neuron_ids,
        input_type: str = 'tuple',
        ):
        '''
        Convert a list of neuron ids to a list of unique identifiers.

        Parameters
        ----------
        neuron_ids: list[tuple[int]] or dataframe['id','subdivision']
            List of neuron ids to convert.
        input_type: str
            Type of the input list.
            Can be 'tuple', 'table'
        '''
        if input_type == 'tuple':
            tuple_of_ids = neuron_ids
        elif input_type == 'table':
            tuple_of_ids = [tuple(row) for row in neuron_ids]
        else:
            raise ValueError(
                f"Class Connections \
                ::: > convert_neuron_ids_to_uid(): Unknown input type {input_type}"
                )
        uid_table = self.uid.copy()
        uid_table = uid_table[uid_table['neuron_ids'].isin(tuple_of_ids)]
        # sort the table by the order of the input list
        uid_table = uid_table.set_index('neuron_ids').loc[tuple_of_ids].reset_index()
        uids = uid_table['uid'].to_list()
        return uids

    def __convert_uid_to_neuron_ids(
        self,
        uids,
        output_type: str = 'tuple',
        ):
        '''
        Convert a list of unique identifiers to a list of neuron ids.

        Parameters
        ----------
        uids: list[int]
            List of unique identifiers to convert.
        output_type: str
            Type of the output list.
            Can be 'tuple', 'table' (dataframe), 'body_id' (only the body ids),
            'subdivision' (only the subdivisions).
        '''
        if not isinstance(uids, list):
            uids = [uids]
        if not output_type in ['tuple','table','body_id','subdivision']:
            raise ValueError(
                f"Class Connections \
                ::: > convert_uid_to_neuron_ids(): Unknown output type {output_type}"
                )
        
        tuple_of_ids = []
        for uid in uids:
            neuron_id = self.uid.loc[
                (self.uid['uid'] == uid)
                ]['neuron_ids'].values[0]
            tuple_of_ids.append(neuron_id)
        if output_type == 'tuple':
            return tuple_of_ids
        elif output_type == 'table':
            return pd.DataFrame(tuple_of_ids, columns = ['id','subdivision'])
        elif output_type == 'body_id':
            return [neuron_id[0] for neuron_id in tuple_of_ids]
        elif output_type == 'subdivision':
            return [neuron_id[1] for neuron_id in tuple_of_ids]
    
    def __get_uids_from_bodyids(self, body_ids: list[int]):
        '''
        Get the unique identifiers from a list of body ids.
        '''
        return self.uid.loc[self.uid['body_id'].isin(body_ids)]['uid'].to_list()
    
    def __build_graph(self): # networkx graph
        self.graph = nx.from_pandas_edgelist(
                    self.connections,
                    source="start_uid",
                    target="end_uid",
                    edge_attr=[
                        "syn_count", # absolute synapse count
                        "predictedNt:string",
                        "eff_weight", # signed synapse count (nt weighted)
                        "syn_count_norm", # input normalized synapse count
                        "eff_weight_norm", # input normalized signed synapse count
                    ],
                    create_using=nx.DiGraph,
                )
        # set an edge attribute 'weight' to the signed synapse count, for compatibility with nx functions
        nx.set_edge_attributes(
            self.graph,
            {
                (u, v): {"weight": d["eff_weight"]}
                for u, v, d in self.graph.edges(data=True)
            }
        )
        # add node attributes
        body_ids = self.__convert_uid_to_neuron_ids(
            self.graph.nodes,
            output_type = 'body_id'
            )
        nx.set_node_attributes(
            self.graph,
            {node: body_id for node, body_id in zip(self.graph.nodes, body_ids)},
            "body_id"
            )
        _ = self.__get_node_attributes("systematicType:string")
        _ = self.__get_node_attributes("class:string")
        nx.set_node_attributes(
            self.graph,
            nx.get_node_attributes(self.graph, "class:string"),
            "node_class"
            )
        nx.set_node_attributes(
            self.graph,
            nx.get_node_attributes(self.graph, "systematicType:string"),
            "node_label"
            )
        return

    def __build_adjacency_matrices(self, nodelist: list[int] = None):
        '''
        Create a dictionary with relevant connectivity matrices
        and the index ordering table.
        '''
        # get only the nodes that have a connection n the graph
        if nodelist is None:
            nodelist = self.uid['uid'].to_list()
            
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
                "body_id": self.__convert_uid_to_neuron_ids(nodelist, output_type='body_id'),
                "subdivision": self.__convert_uid_to_neuron_ids(nodelist, output_type='subdivision'),
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
    
    def __get_node_attributes(self, attribute: str, default='None'):
        '''
        Get the node attributes for the graph.
        If the attribute is not present, identify it in the nodes dataframe and add it to the graph.
        The attributes are then defined based on the initial neuron from which it is defined.
        For instance if a neuron A is subdivided in A1 and A2, the neurotransmitter type 
        of (A,1) and (A,2) will be the same as A.
        '''
        if attribute not in self.graph.nodes:
            nodes = self.get_nodes(type='uid') # uids
            body_ids = self.get_nodes(type='body_id')
            attr = get_nodes_data.load_data_neuron_set( # retrieve data
                ids = body_ids,
                attributes = [attribute],
                )
            attr_mapping = {body_id: value # map body id to attribute value
                        for body_id, value
                        in zip(attr[":ID(Body-ID)"], attr[attribute])}
            attr_list = {node: attr_mapping[body_id] # map node to attribute value
                        for node, body_id in zip(nodes, body_ids)}
            
            # replace None values with a default value
            attr_list = {
                k: default if v is None else v for k, v in attr_list.items()
                }
            nx.set_node_attributes(self.graph, attr_list, attribute)
        return nx.get_node_attributes(self.graph, attribute)
    

    # public methods
    # --- initialize
    def initialize(self, split_neurons: list[Neuron] = None):
        self.__get_connections()
        self.__split_neurons_in_connections(split_neurons)
        self.__map_uid()
        self.__build_graph()
        self.__build_adjacency_matrices()
        print("Connections initialized.")
        return
    
    # --- copy
    def subgraph(self, nodes: list[int] = None, edges: list[tuple[int]] = None):
        '''
        Copy operator of the Connections class that returns a new object
        generated from the subgraph of the current object. 
        All existing connections between the set nodes are included.

        Parameters
        ----------
        nodes: list[int]
            List of nodes to consider in the subgraph.
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
        '''
        # Initialize the new object
        neurons_pre_ = list(set(self.neurons_pre).intersection(set(nodes)))
        neurons_post_ = list(set(self.neurons_post).intersection(set(nodes)))
        subgraph_  = Connections(neurons_pre_, neurons_post_)
        # Set the connections
        connections_ = self.get_connections()
        connections_ = connections_[
            connections_[":START_ID(Body-ID)"].isin(nodes)
            & connections_[":END_ID(Body-ID)"].isin(nodes)
            ]
        if not edges is None:
            rows_to_drop = []
            for index, row in connections_.iterrows():
                if (row[":START_ID(Body-ID)"],
                    row[":END_ID(Body-ID)"]) not in edges:
                    rows_to_drop.append(index)
            connections_ = connections_.drop(rows_to_drop)
        subgraph_.set_connections(connections_)
        # Initialize the graph
        subgraph_.set_graph()
        # Initialize the adjacency matrices
        subgraph_.set_adjacency_matrices()
        return subgraph_

    # --- getters
    def get_neuron_bodyids(self, selection_dict: dict = None) -> list[int]:
        '''
        Get the neuron Body-IDs from the nodes dataframe based on a selection dictionary.
        '''
        nodes = self.get_nodes(type='body_id')
        return get_nodes_data.get_neuron_bodyids(selection_dict, nodes)
    
    def get_neuron_ids(self, selection_dict: dict = None) -> list[int]:
        '''
        Get the neuron IDs from the nodes dataframe based on a selection dictionary.
        '''
        nodes = self.get_nodes(type='body_id')
        body_ids = get_nodes_data.get_neuron_bodyids(selection_dict, nodes)
        return self.__get_uids_from_bodyids(body_ids)

    def get_neurons_pre(self):
        return self.neurons_pre
    
    def get_neurons_post(self):
        return self.neurons_post
    
    def get_connections(self):
        return self.connections
    
    def get_graph(self):
        return self.graph
    
    def get_nt_weights(self):
        return self.nt_weights
    
    def get_adjacency_matrix(
            self,
            type_: str = "syn_count",
            hops: int = 1,
            intermediates: bool = True
        ):
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

    def get_cmatrix(self, type_: str = 'norm'): # cmatrix object
        '''
        Returns the cmatrix object of the connections.
        By default, the normalized adjacency matrix is used.
        '''
        return cmatrix.CMatrix(
            self.get_adjacency_matrix(type_=type_),
            self.get_lookup()
            )

    def get_lookup(self):
        return self.adjacency['lookup']

    def get_nx_graph(
            self,
            hops: int = 1,
            nodes: list[int] = None,
            ):
        '''
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
        '''
        if hops == 1:
            graph_ = self.graph
        else:
            if f'graph_{hops}_hops' not in self.subgraphs.keys():
                self.__compute_n_hops(hops, initialize_graph=True)
            graph_ = self.subgraphs[f'graph_{hops}_hops']
        graph_ = nx_utils.get_subgraph(graph_,nodes)
        return graph_
    
    def get_nodes(self, type: str = 'uid'):
        '''
        Get the nodes of the graph.
        '''
        if type == 'uid':
            return list(self.graph.nodes)
        elif type == 'body_id':
            return list(nx.get_node_attributes(self.graph, "body_id").values())
        else:
            raise ValueError(
                f"Class Connections \
                ::: > get_nodes(): Unknown type {type}"
                )
            
    # --- setters
    def merge_nodes(self, nodes: list[int]):
        '''
        Merge a list of nodes into a single node.
        The first node in the list is kept as reference.
        '''
        ref_node = nodes[0]
        nodes = np.delete(nodes,0)
        for other_node in nodes:
            # merge the nodes in the graph
            self.graph = nx.contracted_nodes(
                self.graph,
                ref_node,
                other_node,
                )
            # merge the nodes in the connections dataframe
            self.connections = self.connections.replace(
                to_replace = other_node,
                value = ref_node,
                )
        # clear the connections table from duplicates: 
        # if a connection is present in both nodes, the weight is summed
        self.connections = self.connections.groupby(
            ["start_uid","end_uid","predictedNt:string"]
            ).agg(
                syn_count = ("syn_count","sum"),
                eff_weight = ("eff_weight","sum"),
                syn_count_norm = ("syn_count_norm","sum"),
                eff_weight_norm = ("eff_weight_norm","sum"),
                ).reset_index()
        
        # update the adjacency matrices
        self.adjacency = self.__build_adjacency_matrices()

        return

    def reorder_neurons(
            self,
            by: str = "list",
            order: list[int] = None,
            order_type: str = "uid",
            ):
        '''
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
            Can be 'body_id' for neuron IDs or 'index' for the adjacency matrix index.
        '''
        match by:
            case "list":
                if order_type == "uid":
                    order_ = order
                elif order_type == "index":
                    order_ = self.adjacency['lookup'].loc[order, 'uid'].to_list()
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
    
    def set_graph(self, graph: nx.DiGraph = None):
        if graph is None:
            graph = self.__build_graph()
        self.graph = graph
        return
    
    def set_nt_weights(self, nt_weights: dict[str, int]):
        self.nt_weights = nt_weights
        return
    
    def set_adjacency_matrices(self):
        self.__build_adjacency_matrices()
        return

    def include_node_attributes(self, attributes: list[str]):
        '''
        Include node attributes in the graph.
        '''
        for attribute in attributes:
            _ = self.__get_node_attributes(attribute)
        return
   
   # --- computations
    def __compute_n_hops(self, n: int, initialize_graph: bool = False):
        '''
        Compute the n-hop adjacency matrix and matching effctive connections graph.
        '''
        #TODO: edit such that it creates a new Connections object instead
        # compute the n-hop adjacency matrix
        self.adjacency[f"mat_norm_{n}_hops"] = matrix_utils.connections_up_to_n_hops(
            self.adjacency["mat_norm"], n
        )
        # create the associated graph
        if initialize_graph:
            self.subgraphs[f'graph_{n}_hops'] = nx.from_scipy_sparse_array(
                self.adjacency[f"mat_norm_{n}_hops"],
                create_using=nx.DiGraph,
            )
            nodes_data_ = dict(self.graph.nodes(data=True))
            nx.set_node_attributes(
                self.subgraphs[f'graph_{n}_hops'],
                nodes_data_,
            )
        return
    
    def paths_length_n(self, n: int, source: list[int], target: list[int]):
        '''
        Get the graph with all the paths of length n between two sets of nodes.

        Parameters
        ----------
        n: int
            Length of the paths.
        source: list[int]
            List of source nodes.
        target: list[int]
            List of target nodes.

        Returns
        -------
        nx.DiGraph
            Graph composed of all the nodes reached by the paths of length n
            between the source and target nodes.
        '''
        if isinstance(source, int):
            source = [source]
        if isinstance(target, int):
            target = [target]
        edges_ = []
        for s in source:
            subedges_ = nx.all_simple_edge_paths(
                self.graph,
                source=s,
                target=target,
                cutoff=n
                )
            _ = [edges_.extend(path) for path in subedges_]
        return nx_utils.get_subgraph_from_edges(self.graph, edges_)
        
    def cluster_hierarchical(self, reference: str = "syn_count"):
        '''
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
        '''
        mat = self.get_adjacency_matrix(reference)
        ordered_mat_, order_ = matrix_utils.cluster_matrix_hierarchical(mat)
        self.reorder_neurons(by="list", order=order_, order_type="index")
        return ordered_mat_


    # --- display
    def display_adjacency_matrix(
            self,
            type_: str = "syn_count",
            method:str = "spy",
            title:str = "test"
        ):
        '''
        Display the adjacency matrix.
        '''
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
                    #xticklabels=False,
                    #yticklabels=False,
                )
            case _:
                raise ValueError(
                    f"Class Connections::: > display_adjacency_matrix(): Unknown method {method}"
                    )

        plt.savefig(os.path.join(params.PLOT_DIR, title + "_matrix.pdf"))
        return
    
    def display_graph(self, method: str = "circular", title: str = "test"):
        '''
        Display the graph.
        '''
        _, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        match method:
            case "circular":
                pos = nx.circular_layout(self.graph)
            case "spring":
                pos = nx.spring_layout(self.graph)
            case _:
                raise ValueError(
                    f"Class Connections > display_graph(): Unknown method {method}"
                    )
        ax = nx_design.draw_graph(
            self.graph, pos, ax=ax, return_pos=False
        )
        plt.savefig(os.path.join(params.PLOT_DIR, title + "_graph.pdf"))
        return 
    
    def display_graph_per_attribute(
            self,
            attribute = 'exitNerve:string',
            center = None ,
            title:str = "test",
            ):
        '''
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
        '''
        # ensures the attribute is present in the graph
        _ = self.__get_node_attributes(attribute) 
        #  use the nx library to plot the graph grouped by the attribute
        if center is not None:
            if isinstance(center, str):
                _ = nx_design.draw_graph_grouped_by_attribute(
                    self.graph,
                    attribute,
                    center_instance=center,
                    )
            elif isinstance(center, int):
                center = [center]
                _ = nx_design.draw_graph_grouped_by_attribute(
                    self.graph,
                    attribute,
                    center_nodes=center,
                    )
            else:
                center = list(center)
                _ = nx_design.draw_graph_grouped_by_attribute(
                    self.graph,
                    attribute,
                    center_nodes=center,
                    )
        
        plt.savefig(os.path.join(params.PLOT_DIR, title + "_sorted_graph.pdf"))

    def plot_xyz(
            self,
            x: list[int],
            y: list[int],
            sorting: str = None,
            title:str = "test",
            ):
        '''
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
        '''
        graph_ = self.get_graph()
        # ensure that the sorting parameter is in the nodes of the graph
        if sorting not in [
            None,'degree','betweenness','from_index','input_clustering',
            'centrality','output_clustering'
            ]:
            _ = self.__get_node_attributes(sorting)

        if sorting == 'from_index':
            z = [node for node in self.graph.nodes
                if not node in x and not node in y]
            sort_list_nodes = self.adjacency['lookup']['uid'].to_list()
            # reorder x,y,z by the order of the adjacency matrix indexings
            x_ = [n for n in sort_list_nodes if n in x]
            y_ = [n for n in sort_list_nodes if n in y]
            z_ = [n for n in sort_list_nodes if n in z]
            pos = nx_design.position_3d_nodes(x_,y_,z_)
            # draw the plot
            _ = nx_design.plot_xyz(graph_,x,y,z=z,sorting=sorting,pos=pos)
        else:
            # restrict x and y to the nodes in the graph
            x = [n for n in x if n in graph_.nodes]
            y = [n for n in y if n in graph_.nodes]
            # draw the plot
            _ = nx_design.plot_xyz(graph_,x,y,sorting=sorting)
        # save the plot
        plt.savefig(os.path.join(params.PLOT_DIR, title + "3d_plot.pdf"))
        return

    def draw_3d_custom_axis(
        self,
        x: list[int],
        y: list[int],
        x_sorting: str = 'degree',
        y_sorting: str = 'exitNerve:string', 
        z_sorting: str = 'input_clustering',
        title: str = "test",
        ):
        '''
        Draw a 3D plot of the connections between two sets of neurons.
        Each axis gets its own sorting parameter.
        '''
        # ensure that the sorting parameters are in the nodes of the graph
        for sorting_method in [x_sorting, y_sorting, z_sorting]:
            if sorting_method not in [
                None,'degree','betweenness','from_index','input_clustering',
                'centrality', 'output_clustering'
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

