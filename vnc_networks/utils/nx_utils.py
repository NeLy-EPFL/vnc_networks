'''
Module containing utility functions for processing nx graphs.
'''
import networkx as nx

def remove_inhibitory_connections(graph: nx.DiGraph):
    """
    Remove inhibitory connections from the graph.
    """
    graph_ = graph.copy()
    edges_to_remove = []
    for edge in graph_.edges():
        if graph_.edges[edge]["weight"] < 0:
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
        if graph_.edges[edge]["weight"] > 0:
            edges_to_remove.append(edge)
    graph_.remove_edges_from(edges_to_remove)
    return graph_

def get_subgraph(graph: nx.DiGraph, nodes: list):
    """
    Get the subgraph of the graph containing only the nodes in the list.
    """
    graph_ = graph.copy()
    return graph_.subgraph(nodes)

def get_subgraph_from_edges(graph: nx.DiGraph, edges: list):
    """
    Get the subgraph of the graph containing only the nodes in the list.
    """
    graph_ = graph.edge_subgraph(edges).copy()
    # remove edges that have a weight of 0
    edges_to_remove = []
    for edge in graph_.edges():
        if abs(graph_.edges[edge]["weight"]) < 1:
            edges_to_remove.append(edge)
    graph_.remove_edges_from(edges_to_remove)
    # nodes that are not connected to any other node are removed
    nodes_to_remove = []
    for node in graph_.nodes():
        if graph_.degree(node) == 0:
            nodes_to_remove.append(node)
    graph_.remove_nodes_from(nodes_to_remove)
    return graph_

def sort_nodes(
        graph: nx.DiGraph,
        nodes: list, 
        sorting: str = None,
        ref: list = None
    ):
    """
    Sort the nodes according to the sorting parameter.
    """
    if nodes is None or len(nodes) == 0:
        return
    # sort the nodes
    match sorting:
        case None:
            return nodes
        case 'alphabetical':
            nodes = sorted(nodes)
        case 'degree':
            nodes = sorted(
                nodes, key=lambda nodes: graph.degree()[nodes]
                )
            nodes = nodes[::-1]
        case 'betweenness':
            nodes = sorted(
                nodes,
                key=lambda nodes: nx.betweenness_centrality(
                    graph,
                    normalized=True
                    )[nodes]
                )
        case 'centrality':
            nodes = sorted(
                nodes,
                key=lambda nodes: nx.closeness_centrality(
                    graph,
                    u=nodes,
                    distance='syn_count',
                    )
                )
        case 'input_clustering':
            new_x = []
            # cluster and order the nodes in x based on their input connections
            for input_node in ref:
                downstream_nodes = nx.descendants_at_distance(
                    graph,
                    source = input_node,
                    distance = 1,
                    )
                target_nodes = downstream_nodes.intersection(set(nodes))
                new_nodes = list(target_nodes.difference(set(new_x)))

                # when ambigous, sort according to the output connections
                if len(new_nodes) > 1:
                    new_nodes = sorted(
                        new_nodes,
                        key=lambda nodes: graph.out_degree()[nodes]
                        )
                # add the new nodes to the list
                new_x.extend(new_nodes)
            # add the remaining nodes
            new_x.extend(list(set(nodes).difference(set(new_x))))
            nodes = new_x
        case 'output_clustering':
            new_x = []
            # cluster and order the nodes in x based on their output connections
            for input_node in ref:
                upstream_nodes, _ = nx.dijkstra_predecessor_and_distance(
                    graph,
                    source = input_node,
                    cutoff = 1,
                    weight = 'syn_count'
                    )
                upstream_nodes = set(
                    [k for k,d in upstream_nodes.items() if len(d) > 0]
                    )
                target_nodes = upstream_nodes.intersection(set(nodes))
                new_nodes = list(target_nodes.difference(set(new_x)))

                # when ambigous, sort according to the output connections
                if len(new_nodes) > 1:
                    new_nodes = sorted(
                        new_nodes,
                        key=lambda nodes: graph.in_degree()[nodes]
                        )
                # add the new nodes to the list
                new_x.extend(new_nodes)
            # add the remaining nodes
            new_x.extend(list(set(nodes).difference(set(new_x))))
            nodes = new_x
        case _:
            nodes = sorted(nodes, key=lambda nodes: graph.nodes[nodes][sorting])

    return nodes
    
def threshold_graph(graph: nx.DiGraph, threshold: int = None):
    """
    Threshold the graph by removing edges with a weight below the threshold.
    """
    graph_ = graph.copy()
    if threshold is None:
        return graph_
    else:
        edges_to_remove = []
        for edge in graph_.edges():
            if abs(graph_.edges[edge]["weight"]) < threshold:
                edges_to_remove.append(edge)
        graph_.remove_edges_from(edges_to_remove)
        # clean up nodes that are not connected to any other node
        nodes_to_remove = []
        for node in graph_.nodes():
            if graph_.degree(node) == 0:
                nodes_to_remove.append(node)
        graph_.remove_nodes_from(nodes_to_remove)
        return graph_