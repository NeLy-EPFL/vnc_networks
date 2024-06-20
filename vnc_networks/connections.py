'''
Initializes the connections class.
Possible use cases are processing the adjacency matrix from a set of neurons,
or displaying the connections between neurons in a graph.
Useful class when juggeling between different representations of the connectome.

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
            self.neurons_pre = neurons_[':ID(Body-ID)'].to_list()
        else:
            self.neurons_pre = neurons_pre
        self.neurons_post = self.neurons_pre if neurons_post is None else neurons_post
        self.nt_weights = params.NT_WEIGHTS
        self.subgraphs = {}
          

    # private methods
    def __get_connections(self):
        connections_ = pd.read_feather(params.NEUPRINT_CONNECTIONS_FILE)
        # filter out only the connections relevant here
        connections_ = connections_[
            connections_[":START_ID(Body-ID)"].isin(self.neurons_pre)
            & connections_[":END_ID(Body-ID)"].isin(self.neurons_post)
            ]
        
        # rename synapse count column explicitly and filter
        connections_["syn_count"] = connections_["weight:int"]
        connections_ = connections_[
            connections_["syn_count"] >= params.SYNAPSE_CUTOFF
            ]
        
        # add relevant information for the processing steps

        ## add the neurotransmitter type to the connections
        nttypes = get_nodes_data.load_data_neuron_set(
            self.neurons_pre,["predictedNt:string"]
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
        connections_["syn_count_norm"] = connections_["syn_count"] / connections_["in_total"]
        connections_["eff_weight_norm"] = connections_["eff_weight"] / connections_["in_total"]
        connections_ = connections_.drop(columns=["in_total"])

        return connections_

    def __build_graph(self): # networkx graph
        graph_ = nx.from_pandas_edgelist(
                    self.connections,
                    source=":START_ID(Body-ID)",
                    target=":END_ID(Body-ID)",
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
            graph_,
            {
                (u, v): {"weight": d["eff_weight"]}
                for u, v, d in graph_.edges(data=True)
            }
        )
        # add node attributes
        node_names = get_nodes_data.load_data_neuron_set(
            ids = list(graph_.nodes),
            attributes = ["systematicType:string","class:string"],
            )
        names = {node_id: name 
                for node_id, name in zip(node_names[":ID(Body-ID)"],
                                        node_names["systematicType:string"]
                                        )
                }
        class_ = {node_id: name
                for node_id, name in zip(node_names[":ID(Body-ID)"],
                                        node_names["class:string"]
                                        )
                }
        nx.set_node_attributes(graph_, names, "node_label")
        nx.set_node_attributes(graph_, class_, "node_class")
        return graph_

    def __build_adjacency_matrices(self, nodelist: list[int] = None):
        '''
        Create a dictionary with relevant connectivity matrices
        and the index ordering table.
        '''
        # get only the nodes that have a connection n the graph
        if nodelist is None:
            nodelist = list(self.neurons_pre) + list(
                set(self.neurons_post) - set(self.neurons_pre) # avoid duplicates
                )
            
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
            data={"index": np.arange(len(nodelist)), "body_id": nodelist}
        )
        lookup = lookup.sort_values("index")

        return {
            "mat_norm": mat_norm,
            "mat_unnorm": mat_unnorm,
            "mat_syncount": mat_syn_count,
            "lookup": lookup,
        }
    
    def __get_node_attributes(self, attribute: str, default = 'None'):
        '''
        Get the node attributes for the graph. If the attribute is not present,
        identify it in the nodes dataframe and add it to the graph.
        '''
        if attribute not in self.graph.nodes:
            attr = get_nodes_data.load_data_neuron_set(
                ids = self.graph.nodes,
                attributes = [attribute],
                )
            attr_list = {node_id: value
                        for node_id, value
                        in zip(attr[":ID(Body-ID)"], attr[attribute])}
            # replace None values with a default value
            attr_list = {
                k: default if v is None else v for k, v in attr_list.items()
                }
            nx.set_node_attributes(self.graph, attr_list, attribute)
        return nx.get_node_attributes(self.graph, attribute)
    

    # public methods
    # --- initialize
    def initialize(self):
        self.connections = self.__get_connections()
        self.graph = self.__build_graph()
        self.adjacency = self.__build_adjacency_matrices()
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
    def get_neuron_ids(self, selection_dict: dict = None) -> list[int]:
        '''
        Get the neuron IDs from the nodes dataframe based on a selection dictionary.
        '''
        nodes = self.get_nodes()
        return get_nodes_data.get_neuron_ids(selection_dict, nodes)

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

    def get_cmatrix(self, type: str = 'norm'): # cmatrix object
        '''
        Returns the cmatrix object of the connections.
        By default, the normalized adjacency matrix is used.
        '''
        return cmatrix.Cmatrix(
            self.get_adjacency_matrix(type_=type),
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
    
    def get_nodes(self):
        return self.get_graph().nodes()
    
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
            [":START_ID(Body-ID)",":END_ID(Body-ID)","predictedNt:string"]
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
            order_type: str = "body_id",
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
                if order_type == "body_id":
                    order_ = order
                elif order_type == "index":
                    order_ = self.adjacency['lookup'].loc[order, 'body_id'].to_list()
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
        self.adjacency = self.__build_adjacency_matrices()
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
    
    def display_graph(self, method: str = "circular", title:str = "test"):
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
            sort_list_nodes = self.adjacency['lookup']['body_id'].to_list()
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

