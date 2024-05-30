'''
Initializes the connections class.
Possible use cases are processing the adjacency matrix from a set of neurons,
or displaying the connections between neurons in a graph.

Use the following code to initialize the class:
```
from connections import Connections
neurons_pre = get_neurons_from_class('sensory neuron')
neurons_post = get_neurons_from_class('motor neuron')
connections = Connections(neurons_pre, neurons_post)
# connections.set_nt_weights({'ACH': +1, 'GABA': -1, 'GLUT': -1, 'SER': 0, 'DA': 0, 'OCT': 0})
connections.initialize()
``` 
'''
import pandas as pd
import numpy as np
import networkx as nx

import params
from get_nodes_data import load_data_neuron_set

class Connections:
    def __init__(self, neurons_pre: list[int], neurons_post: list[int] = None):
        self.neurons_pre = neurons_pre
        self.neurons_post = neurons_post if neurons_post is not None else neurons_pre
        self.nt_weights = params.NT_WEIGHTS

    # private methods
    def initialize(self):
        self.connections = self.__get_connections()
        self.graph = self.__build_graph()
        self.adjacency = self.__build_adjacency_matrices()

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
        nttypes = load_data_neuron_set(self.neurons_pre,["predictedNt:string"])
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
                        "syn_count",
                        "predictedNt:string",
                        "eff_weight",
                        "syn_count_norm",
                        "eff_weight_norm",
                    ],
                    create_using=nx.DiGraph,
                )
        # TODO: add node and edge attributes as convenient (e.g. name, location etc.)
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

        return {
            "mat_norm": mat_norm,
            "mat_unnorm": mat_unnorm,
            "mat_syncount": mat_syn_count,
            "lookup": lookup,
        }
       
        


    # public methods
    # --- getters
    def get_neurons_pre(self):
        return self.neurons_pre
    
    def get_neurons_post(self):
        return self.neurons_post
    
    def get_adjacency_matrix(self, type_: str = "syn_count"):
        match type_:
            case "norm":
                return self.adjacency["mat_norm"]
            case "unnorm":
                return self.adjacency["mat_unnorm"]
            case "syn_count":
                return self.adjacency["mat_syncount"]
            case _:
                raise ValueError(f"Class Connections::: > get_adjacency_matrix(): Unknown type {type_}")

    def get_graph(self):
        return self.graph
    
    # --- setters
    def reorder_neurons(self, by: str = "list", order: list[int] = None):
        '''
        Reorder the neurons in the adjacency matrices and the corresponding lookup table.
        '''
        match by:
            case "list":
                self.adjacency = self.__build_adjacency_matrices(nodelist=order)
            case _:
                raise ValueError(f"Class Connections::: > reorder_neurons(): Unknown method {by}")
        return

    # --- display
    def display_adjacency_matrix(self, type_: str = "syn_count", method:str = "spy", title:str = "test"):
        '''
        Display the adjacency matrix.
        '''
        import matplotlib.pyplot as plt
        import os
        
        plt.figure(figsize=(8, 8))
        match method:
            case "spy":
                plt.spy(self.get_adjacency_matrix(type_), markersize=0.1)
            case "heatmap":
                import seaborn as sns
                mat = self.get_adjacency_matrix(type_)
                sns.heatmap(mat.toarray(), cmap="flare")
            case _:
                raise ValueError(f"Class Connections::: > display_adjacency_matrix(): Unknown method {method}")

        plt.savefig(os.path.join(params.PLOT_DIR, title + ".pdf"))
    
