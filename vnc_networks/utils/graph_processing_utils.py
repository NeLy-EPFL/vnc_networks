'''
Module containing utility functions for processing nx graphs.
'''


import networkx as nx

def remove_inhbitory_connections(graph: nx.DiGraph):
    """
    Remove inhibitory connections from the graph.
    """
    graph_ = graph.copy()
    edges_to_remove = []
    for edge in graph_.edges():
        if graph_.edges[edge]["eff_weight"] < 0:
            edges_to_remove.append(edge)
    graph_.remove_edges_from(edges_to_remove)
    return graph_

def remove_excitatory_connections(graph: nx.DiGraph):
    """
    Remove excitatory connections from the graph.
    """
    graph_ = graph.copy()
    edges_to_remove = []
    for edge in graph_.edges():
        if graph_.edges[edge]["eff_weight"] > 0:
            edges_to_remove.append(edge)
    graph_.remove_edges_from(edges_to_remove)
    return graph_
    