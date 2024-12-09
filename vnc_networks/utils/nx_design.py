'''
Helper functions for making networkx graphs look nice and standardized.
'''
import typing
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import Counter
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.axes
from typing import Optional


import params
import utils.nx_utils as nx_utils
from utils.math_utils import sigmoid
import params

#--- Constants ---#


#--- CODE ---#

@typing.overload
def draw_graph(
    G: nx.Graph | nx.DiGraph,
    pos: Optional[typing.Mapping] = None,
    pos_nx_method=nx.circular_layout,
    ax: Optional[matplotlib.axes.Axes] = None,
    node_size: int = params.NODE_SIZE,
    return_pos: typing.Literal[False] = False,
    label_nodes: bool = False,
    add_legend: bool = True,
    connection_style="arc3,rad=0.1",
) -> matplotlib.axes.Axes: ...
@typing.overload
def draw_graph(
    G: nx.Graph | nx.DiGraph,
    pos: Optional[typing.Mapping] = None,
    pos_nx_method=nx.circular_layout,
    ax: Optional[matplotlib.axes.Axes] = None,
    node_size: int = params.NODE_SIZE,
    return_pos: typing.Literal[True] = True,
    label_nodes: bool = False,
    add_legend: bool = True,
    connection_style="arc3,rad=0.1",
) -> tuple[matplotlib.axes.Axes, dict]: ...

def draw_graph(
    G: nx.Graph | nx.DiGraph,
    pos: Optional[typing.Mapping] = None,
    pos_nx_method=nx.circular_layout,
    ax: Optional[matplotlib.axes.Axes] = None,
    node_size: int = params.NODE_SIZE,
    return_pos: bool = False,
    label_nodes: bool = False,
    add_legend: bool = True,
    connection_style="arc3,rad=0.1",
) -> matplotlib.axes.Axes | tuple[matplotlib.axes.Axes, dict]:
    """Plots the network using the network specs and
    returns the positions of the nodes.

    Parameters
    ----------
    G : nx.Graph
        Networkx graph object.
    ax : plt.Axes, optional
        Matplotlib axes object, by default None
    pos : dict, optional
        Dictionary containing the positions of the nodes, by default None
    pos_nx_method : nx.layout, optional
        Method to use for the node positioning, by default nx.circular_layout
    node_size : int, optional
        Size of the nodes, by default as defined in the constants

    Returns
    -------
    pos: Dict, optional
        Dictionary containing node ID as keys and positions of nodes
        on the figure as values.
    ax: plt.Axes
    """

    if ax is None:
        _, ax = plt.subplots(figsize=params.FIGSIZE, dpi=params.DPI)
    assert ax is not None # needed for type hinting

    if pos is None:
        if pos_nx_method == nx.kamada_kawai_layout:
            # method not defined on negative edge weights
            # add a 'pos_weight' attribute to the graph
            for u, v, data in G.edges(data=True):
                data['pos_weight'] = np.abs(data['weight'])
            pos = nx.kamada_kawai_layout(G, weight='pos_weight')
        else:
            pos = pos_nx_method(G)

    # Normalize the edge width of the network
    weights = list(nx.get_edge_attributes(G, "weight").values())
    normalized_weights = (
        np.abs(weights) / np.abs(weights).max()
        ) * params.MAX_EDGE_WIDTH

    # Color nodes and edges
    node_colors = define_node_colors(G)
    edge_colors = define_edge_colors(G)

    # Node labels
    labels = nx.get_node_attributes(G, "node_label")
    node_labels = {
        k: labels[k] if k in labels.keys() else k 
        for k in G.nodes
        }

    # Plot graph
    nx.draw(
        G,
        pos,
        with_labels=label_nodes,
        labels=node_labels,
        width=normalized_weights,
        alpha=0.75,
        node_size=node_size,
        node_color=node_colors,
        edge_color=edge_colors,
        connectionstyle=connection_style,
        font_size=params.FONT_SIZE,
        font_color=params.FONT_COLOR,
        ax=ax,
    )

    # Add legend
    if add_legend:
        ax = add_edge_legend(
                ax,
                normalized_weights,
                edge_colors,
                params.MAX_EDGE_WIDTH/np.abs(weights).max(),
                )

    # return
    if return_pos:
        return ax, pos
    return ax

def add_edge_legend(
    ax: matplotlib.axes.Axes,
    weights: list,
    color_list: list,
    arrow_norm: float,
):
    """
    Add a legend to the plot with the edge weights.
    """
    color = Counter(color_list).most_common(1)[0][0]
    lines = []
    edges_weight_list = sorted(
        [np.abs(w) for w in weights if w != 0], reverse=True
    )
    for _, width in enumerate(edges_weight_list):
        lines.append(Line2D([], [], linewidth=width, color=color))
    # keep only 3 legend entries, evenly spaced, including the first and last
    if len(edges_weight_list) > 3:
        edges_weight_list = [
            edges_weight_list[0],
            edges_weight_list[len(edges_weight_list) // 2],
            edges_weight_list[-1],
        ]
        lines = [lines[0], lines[len(lines) // 2], lines[-1]]

    edges_weight_list = [
        f"{int(weight/arrow_norm)}" for weight in edges_weight_list
    ]
    ax.legend(
        lines,
        edges_weight_list,
        bbox_to_anchor=(0.1, 0.25),
        frameon=False,
    )
    return ax

def define_edge_colors(graph: nx.DiGraph):
    """
    Define the colors of the edges based on the neurotransmitter type.
    If the neurotransmitter type is not defined, the color is 
    based on the sign of the weight.
    """
    edges = graph.edges(data=True)
    edge_colors = []
    edge_color_dict = {}
    for e_ in edges:
        if "edge_color" in e_[2].keys():
            edge_colors.append(e_[2]["edge_color"])
        else:
            if "predictedNt:string" in e_[2].keys():
                edge_colors.append(
                    params.NT_TYPES[e_[2]["predictedNt:string"]]["color"]
                    )
            else:
                if not "weight" in e_[2].keys():
                    edge_colors.append(params.LIGHTGREY)
                elif e_[2]["weight"] >= 0:
                    edge_colors.append(params.EXCIT_COLOR)
                else:
                    edge_colors.append(params.INHIB_COLOR)
        edge_color_dict[(e_[0], e_[1])] = {"edge_color": edge_colors[-1]}
    graph = nx.set_edge_attributes(graph, values=edge_color_dict)
    return edge_colors

def define_node_colors(graph: nx.DiGraph):
    """
    Define the colors of the nodes based on the explicit definition.
    If the color is not defined, the color is based on the class.
    """
    nodes_ = graph.nodes(data=True)
    node_colors = []
    node_dict = {}
    for (n_,dict_) in nodes_:
        if "node_color" in dict_.keys():
            node_colors.append(dict_["node_color"])
        else:
            if "color" in dict_.keys():
                node_colors.append(dict_["color"])
            elif "node_class" in dict_.keys():
                node_colors.append(
                    params.NEURON_CLASSES[dict_["node_class"]]["color"])
            else:
                node_colors.append(params.LIGHTGREY)
        node_dict[n_] = {"node_color": node_colors[-1]}
    graph = nx.set_node_attributes(graph, values=node_dict)
    return node_colors
 
def define_node_boundary_colors(graph: nx.DiGraph):
    """
    Define the colors of the nodes based on their neurotransmitter type.
    """
    nodes_ = graph.nodes(data=True)
    node_b_colors = []
    for (n_,dict_) in nodes_:
        if "boundary_color" in dict_.keys():
            node_b_colors.append(dict_["boundary_color"])
        elif "predictedNt:string" in dict_.keys():
            node_b_colors.append(
                params.NT_TYPES[n_["predictedNt:string"]]["color"]
                )
        else:
            node_b_colors.append(params.WHITE)
    graph = nx.set_node_attributes(
        graph, values=node_b_colors, name="boundary_color"
        )
    return node_b_colors

def get_ring_pos_sorted_alphabetically(
    node_list: list,
    graph: nx.DiGraph,
    radius: float = 3,
    center: tuple | None = (0,0),
    ):
    """
    Define the positions of the nodes in a circular layout, sorted alphabetically.
    """
    if center is None:
        center = (0,0)
    subgraph = graph.subgraph(node_list)
    raw_pos = nx.circular_layout(subgraph, scale=radius, center=center)
    def get_node_label(node):
        if ("node_label" not in subgraph.nodes[node].keys()
        ) or (subgraph.nodes[node]["node_label"] is None):
            return ''
        return subgraph.nodes[node]["node_label"]
    ordered_nodes = sorted(
        list(subgraph.nodes()),
        key=get_node_label,
    )
    pos = {node_: pos_ for node_,pos_ in zip(ordered_nodes,list(raw_pos.values()))}
    return pos

def draw_graph_concentric_circles(
    graph: nx.DiGraph,
    ax: Optional[matplotlib.axes.Axes] = None,
    edge_norm: Optional[float] = None,
    pos:Optional[dict]=None,
    center:Optional[tuple]=None,
    radius_scaling:float=1,
    output:str='graph'
):
    """
    Draw a graph with the nodes and edges attributes based on the
    nx.draw_networkx function and the properties of the graph itself.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to draw
    ax : Optional[matplotlib.axes.Axes], optional
        The axes to plot on, by default None
    edge_norm : float, optional
        The normalisation factor for the edge width, by default None
    pos : dict, optional
        The positions of the nodes, by default None
    center : tuple, optional
        The center of the graph, by default None
    output : str, optional
        Whether to return the figure 'graph' or the positions 'pos', by default 'graph'
    """
    

    # prune network: if a an edge has weight 0, remove it
    graph_ = graph.copy()
    edges_to_remove = []
    for u, v, data in graph_.edges(data=True):
        if data["weight"] == 0:
            edges_to_remove.append((u, v))
    graph_.remove_edges_from(edges_to_remove)

    # outer ring: nodes that don't output to other nodes
    outer_ring = [
        n for n in graph_.nodes if len(list(graph_.successors(n))) == 0
    ]
    pos_outer = get_ring_pos_sorted_alphabetically(
        outer_ring, graph, radius=radius_scaling, center=center
        )

    # inner ring: nodes that only output to other nodes
    input_only_nodes = [
        n for n in graph.nodes
        if len(list(graph_.predecessors(n))) == 0
        and len(list(graph_.successors(n))) > 0
        ]
    pos_inner = get_ring_pos_sorted_alphabetically(
        input_only_nodes, graph_, radius=radius_scaling*0.5, center=center
        )

    # central ring: nodes that both input and output to other nodes
    central_ring = graph.nodes - set(outer_ring) - set(input_only_nodes)
    pos_central = get_ring_pos_sorted_alphabetically(
        central_ring, graph_, radius=radius_scaling*0.75, center=center
        )

    # merge the positions
    pos = {**pos_outer, **pos_inner, **pos_central}

    # define the colors of the nodes
    node_colors = [
        graph_.nodes[n]["node_color"]
        if "node_color" in graph_.nodes[n].keys()
        else params.LIGHTGREY
        for n in graph_.nodes
    ]

    node_labels = {
        n: graph_.nodes[n]["node_label"]
        if "node_label" in graph_.nodes[n].keys()
        else ""
        for n in graph_.nodes
    }

    # define the colors and widths of the edges based on the weights
    if edge_norm is None:
        try:
            edge_norm = max(
                [np.abs(graph_.edges[e]["weight"]) for e in graph_.edges]
                ) / 5
        except ValueError:
            edge_norm = 1
    widths = [
        np.abs(graph_.edges[e]["weight"]) / edge_norm for e in graph_.edges
        ]
    edge_colors = define_edge_colors(graph_)

    if output == 'graph':
        if ax is None:
            _, ax = plt.subplots(figsize=params.FIGSIZE, dpi=params.DPI)
        # Plot graph
        nx.draw(
            graph_,
            pos,
            nodelist=graph_.nodes,
            with_labels=True,
            labels=node_labels,
            alpha=0.5,
            node_size=100,
            node_color=node_colors,
            edge_color=edge_colors,
            width=widths,
            connectionstyle="arc3,rad=0.1",
            font_size=5,
            font_color="black",
            ax=ax,
        )
        if len(widths) > 0:
            add_edge_legend(
                ax,
                weights=widths,
                color_list=edge_colors,
                arrow_norm=1 / edge_norm,
            )
        return ax
    elif output == 'pos':
        return pos

def draw_graph_grouped_by_attribute(
        graph: nx.DiGraph,
        attribute: str,
        restricted_connections: Optional[str] = None,
        position_reference: Optional[str] = None,
        center_instance: Optional[str] = None,
        center_nodes: Optional[list] = None,
        ):
    """
    The plot resembles a large flower, where each petal is a cluster of neurons.
    Each petal is coloured according to the type of the neurons it contains.
    Each petal is made of three concentric circles, where the neurons are placed.
    The neurons are placed according to their connectiivty within the cluster.
    The inner circle contains only neurons that output to other neurons within the cluster
    without receiving any input from them.
    The middle circle contains neurons that receive inputs and outputs from other neurons within the cluster. 
    The outer circle is the rest (only inputs or no connections within the cluster
    in case we restrict to a subset).
    In the middle of the flower, there is a circle containing the neurons for which the attribute is not defined.
    Alternatively, the center can be occupied by a given predefined instance of the attribute.
    The positions can be defined with regards to a specific type of connection
    (inhibitory or excitatory) or all connections. 

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to plot.
    attribute : str
        The attribute to group the neurons by.
    restricted_connections : str, optional
        The type of connections to consider, by default None
    position_reference : str, optional
        The type of connections to consider for the positions, by default None
    center_instance : str, optional 
        The attribute to place at the center, by default None
    center_nodes : list, optional
        The nodes to place at the center, by default None. Overwrites center_instance.

    Raises
    ------
    ValueError
        If the requested attribute is not defined in the graph.

    Returns
    -------
    plt.Axes
    """
    # verify that the attribute is defined for all nodes.
    # if not, add a default value 'unknown'
    for node in graph.nodes():
        if not attribute in graph.nodes[node].keys() or graph.nodes[node][attribute] is None:
            graph.nodes[node][attribute] = 'None'

    # count the number of clusters defined
    cluster_list = [graph.nodes[node][attribute]
        for node in graph.nodes() if not node in center_nodes]
    clusters = list(set(cluster_list))
    clusters.sort()

    if clusters == []:
        raise ValueError(f'No clustering defined by {attribute} in the graph')
    
    if (center_nodes is None) and (not center_instance is None) and (center_instance in clusters):
        clusters.remove(center_instance) # will be placed in the center afterwards
    nb_clusters = len(clusters)


    cluster_centers = [
        ((0.85*params.FIGSIZE[0])*np.cos(2*np.pi*cluster_nb/nb_clusters),
        (0.85*params.FIGSIZE[1])*np.sin(2*np.pi*cluster_nb/nb_clusters))
        #((1+1.2*nb_clusters)*np.cos(2*np.pi*cluster_nb/nb_clusters),
        #(1+1.2*nb_clusters)*np.sin(2*np.pi*cluster_nb/nb_clusters))
        for cluster_nb in range(nb_clusters)
        ]

    positions = {}
    leaf_scaling = 1 / nb_clusters * params.FIGSIZE[0]

    for cluster_idx, cluster_nb in enumerate(clusters):
        # get a subset of the graph known_graph, where the 'cluster' attribute is equal to cluster_of_interest
        subgraph = graph.subgraph(
            [
                node
                for node in graph.nodes()
                if graph.nodes[node][attribute] == cluster_nb
                and not node in center_nodes
            ]
            )
        # use the network with specific connectivity to define the positions of the neurons
        if position_reference == 'inhibitory': 
            subgraph = nx_utils.remove_excitatory_connections(subgraph)
        elif position_reference == 'excitatory':
            subgraph = nx_utils.remove_inhibitory_connections(subgraph)

        # draw the graph
        pos = draw_graph_concentric_circles(
            subgraph,
            center=cluster_centers[cluster_idx],
            output='pos',
            radius_scaling=leaf_scaling,
            )
        positions = {**positions, **pos}

    # additional nodes in the center of the flower
    if center_nodes is not None:
        additional_nodes = center_nodes
    elif center_instance is not None:
        additional_nodes = [
            node for node in graph.nodes()
            if graph.nodes[node][attribute] == center_instance
            ]
    subgraph = graph.subgraph(additional_nodes)
    if position_reference == 'inhibitory': 
        subgraph = nx_utils.remove_excitatory_connections(subgraph)
    elif position_reference == 'excitatory':
        subgraph = nx_utils.remove_inhibitory_connections(subgraph)
    pos = draw_graph_concentric_circles(
        subgraph,
        center=(0,0),
        output='pos',
        radius_scaling=0.3 * params.FIGSIZE[0] * sigmoid(nb_clusters),
        )
    positions = {**positions, **pos}


    # draw the graph
    fig, ax = plt.subplots(figsize=params.FIGSIZE, dpi=params.DPI)

    if restricted_connections == 'inhibitory': 
        graph = nx_utils.remove_excitatory_connections(graph)
    elif restricted_connections == 'excitatory':
        graph = nx_utils.remove_inhibitory_connections(graph)

    edge_norm = max([np.abs(graph.edges[e]["weight"]) for e in graph.edges]) / 5
    widths = [np.abs(graph.edges[e]["weight"]) / edge_norm for e in graph.edges]
    edge_colors = define_edge_colors(graph)
    node_colors = define_node_colors(graph)
    node_boundary_colors = define_node_boundary_colors(graph)
    node_labels = {
            n: graph.nodes[n]["node_label"]
            if "node_label" in graph.nodes[n].keys()
            else ""
            for n in graph.nodes
    }
    nx.draw(
        graph,
        pos=positions,
        nodelist=graph.nodes,
        with_labels=True,
        labels=node_labels,
        alpha=0.5,
        node_size=params.NODE_SIZE,
        node_color=node_colors,
        #edgecolors = node_boundary_colors, # tODO: test this
        edge_color=edge_colors,
        width=widths,
        connectionstyle="arc3,rad=0.1",
        font_size=2,
        font_color="black",
        ax=ax,
    )
    add_edge_legend(ax,
                    weights=widths,
                    color_list=edge_colors,
                    arrow_norm=1 / edge_norm,)
    
    # add labels to the clusters
    for cluster_idx, cluster_nb in enumerate(clusters):
        ax.text(
            cluster_centers[cluster_idx][0],
            cluster_centers[cluster_idx][1],
            cluster_nb,
            fontsize=10,
            color='k',
        )
    if (center_nodes is None) and (not center_instance is None):
        ax.text(
            0,
            0,
            center_instance,
            fontsize=10,
            color='k',
        )
    return ax


@typing.overload
def draw_graph_concentric_by_attribute(
    graph: nx.DiGraph,
    attribute: str,
    center_nodes: list[int],
    target_nodes: list[int],
    ax: Optional[matplotlib.axes.Axes] = None,
    return_pos: typing.Literal[False] = False,
    label_nodes: bool = False,
    ) -> matplotlib.axes.Axes: ...
@typing.overload
def draw_graph_concentric_by_attribute(
    graph: nx.DiGraph,
    attribute: str,
    center_nodes: list[int],
    target_nodes: list[int],
    ax: Optional[matplotlib.axes.Axes] = None,
    return_pos: typing.Literal[True] = True,
    label_nodes: bool = False,
    ) -> tuple[matplotlib.axes.Axes, dict]: ...
def draw_graph_concentric_by_attribute(
    graph: nx.DiGraph,
    attribute: str,
    center_nodes: list[int],
    target_nodes: list[int],
    ax: Optional[matplotlib.axes.Axes] = None,
    return_pos: bool = False,
    label_nodes: bool = False,
    ) -> matplotlib.axes.Axes | tuple[matplotlib.axes.Axes, dict]:
    '''
    Represent the graph with 3 concentric circles.
    center_nodes are on a center circle, typically input neurons.
    On the outer circle are target_nodes grouped by attribute.
    On the intermediate circle are the rest of the nodes in the graph,
    with their angle minimising distance to the nodes they're connected to
    for readability.
    '''
    # --- positions of the nodes ---
    positions = {}

    # === outer ring: target neurons, grouped by attribute
    outer_radius = 0.85
    # count the number of clusters defined
    cluster_list = [
        graph.nodes[node][attribute]
        for node in target_nodes if node in graph.nodes
        ]
    clusters = list(set(cluster_list))
    clusters.sort()
    if clusters == []:
        raise ValueError(f'No clustering defined by {attribute} in the graph')
    nb_clusters = len(clusters)
    cluster_centers = [
        ((outer_radius*params.FIGSIZE[0])*np.cos(2*np.pi*cluster_nb/nb_clusters),
        (outer_radius*params.FIGSIZE[1])*np.sin(2*np.pi*cluster_nb/nb_clusters))
        for cluster_nb in range(nb_clusters)
        ]
    leaf_scaling = 1 / nb_clusters * params.FIGSIZE[0]
     
    for cluster_idx, cluster_nb in enumerate(clusters):
        # get a subset of the graph known_graph, where the 'cluster' attribute is equal to cluster_of_interest
        subgraph = graph.subgraph(
            [
                node
                for node in target_nodes
                if graph.nodes[node][attribute] == cluster_nb
            ]
            )
        # define positions
        pos = draw_graph_concentric_circles(
            subgraph,
            center=cluster_centers[cluster_idx],
            output='pos',
            radius_scaling=leaf_scaling,
            )
        positions = {**positions, **pos}

    # === middle ring: nodes not in the target_nodes nor in the center_nodes
    middle_radius = 0.4
    # get the nodes that are not in the target_nodes
    other_nodes = [
        node for node in graph.nodes
        if not node in target_nodes and not node in center_nodes
        ]
    # get access to the angles of connected nodes
    def angle_node(node, positions):
        if not node in positions.keys():
            return np.random.rand()*2*np.pi
        return np.arctan2(positions[node][1], positions[node][0])
    # compute the weighted angle
    def weighted_angle(graph, node_to_place, connected_nodes, positions):
        if len(connected_nodes) == 0:
            return np.random.rand()*2*np.pi
        angles = [angle_node(n, positions) for n in connected_nodes]
        weights = [
            np.abs(graph.edges[(node_to_place,n)]['weight'])
            for n in connected_nodes
            ]
        return np.dot(weights, angles) / sum(weights)
    # define the positions
    for node in other_nodes:
        # get the connected nodes
        connected_nodes = list(graph.successors(node))
        # get the angles of the connected nodes
        w_angle = weighted_angle(graph, node, connected_nodes, positions)
        # get the position
        pos = (
            middle_radius*params.FIGSIZE[0]*np.cos(w_angle),
            middle_radius*params.FIGSIZE[1]*np.sin(w_angle)
            )
        positions[node] = pos

    # === inner ring: center nodes
    inner_radius = 0.1
    # define the positions
    for node in center_nodes:
        connected_nodes = list(graph.successors(node))
        w_angle = weighted_angle(graph, node, connected_nodes, positions)
        pos = (
            inner_radius*params.FIGSIZE[0]*np.cos(w_angle),
            inner_radius*params.FIGSIZE[1]*np.sin(w_angle)
            )
        positions[node] = pos

    # --- draw the graph ---
    if ax is None:
        fig, ax = plt.subplots(figsize=params.FIGSIZE, dpi=params.DPI)
    assert ax is not None # needed for type hinting
    edge_norm = max([np.abs(graph.edges[e]["weight"]) for e in graph.edges]) / 5
    widths = [np.abs(graph.edges[e]["weight"]) / edge_norm for e in graph.edges]
    edge_colors = define_edge_colors(graph)
    node_colors = define_node_colors(graph)
    node_labels = {
            n: graph.nodes[n]["node_label"]
            if "node_label" in graph.nodes[n].keys()
            else ""
            for n in graph.nodes
    }
    nx.draw(
        graph,
        pos=positions,
        nodelist=graph.nodes,
        with_labels=label_nodes,
        labels=node_labels,
        alpha=0.5,
        node_size=params.NODE_SIZE,
        node_color=node_colors,
        edge_color=edge_colors,
        width=widths,
        connectionstyle="arc3,rad=0.1",
        font_size=params.FONT_SIZE,
        font_color="black",
        ax=ax,
    )
    add_edge_legend(ax,
                    weights=widths,
                    color_list=edge_colors,
                    arrow_norm=1 / edge_norm,)
    
    # add labels to the clusters
    for cluster_idx, cluster_nb in enumerate(clusters):
        ax.text(
            cluster_centers[cluster_idx][0],
            cluster_centers[cluster_idx][1],
            cluster_nb,
            fontsize=10,
            color='k',
        )

    if return_pos:
        return ax, positions
    return ax   

@typing.overload
def draw_graph_in_out_center_circle(
    graph: nx.DiGraph,
    top_nodes: list[int],
    bottom_nodes: list[int],
    ax: Optional[matplotlib.axes.Axes] = None,
    return_pos: typing.Literal[False] = False,
    label_nodes: bool = False,
    ) -> matplotlib.axes.Axes: ...
@typing.overload
def draw_graph_in_out_center_circle(
    graph: nx.DiGraph,
    top_nodes: list[int],
    bottom_nodes: list[int],
    ax: Optional[matplotlib.axes.Axes] = None,
    return_pos: typing.Literal[True] = True,
    label_nodes: bool = False,
    ) -> tuple[matplotlib.axes.Axes, dict]: ...
def draw_graph_in_out_center_circle(
    graph: nx.DiGraph,
    top_nodes: list[int],
    bottom_nodes: list[int],
    ax: Optional[matplotlib.axes.Axes] = None,
    return_pos: bool = False,
    label_nodes: bool = False,
    ) -> matplotlib.axes.Axes | tuple[matplotlib.axes.Axes, dict]:
    '''
    Represent the graph with an upper and lower rows of nodes,
    and a central circle.
    '''
    # === prepare the display of the nodes ===
    if ax is None:
        fig, ax = plt.subplots(figsize=params.FIGSIZE, dpi=params.DPI)
    assert ax is not None # needed for type hinting

    # === prepare the positions of the nodes ===
    positions = {}

    # --- top row: top_nodes
    # display all the nodes uniformly on the top row as a line
    width = params.FIGSIZE[0]
    height = 0.95*width
    for idx, node in enumerate(top_nodes):
        pos = (width*(idx+1)/(len(top_nodes)+1), height)
        positions[node] = pos

    # --- bottom row: bottom_nodes
    # display all the nodes uniformly on the bottom row as a line
    height = 0.05*width
    for idx, node in enumerate(bottom_nodes):
        pos = (width*(idx+1)/(len(bottom_nodes)+1), height)
        positions[node] = pos

    # --- central circle: the rest of the nodes
    other_nodes = [
        node for node in graph.nodes
        if not node in top_nodes and not node in bottom_nodes
        ]
    for idx, node in enumerate(other_nodes):
        angle = 2*np.pi*idx/len(other_nodes)
        pos = (
            width*(0.5 + 0.25*np.cos(angle)),
            width*(0.5 + 0.25*np.sin(angle))
            )
        positions[node] = pos

    # === draw the graph ===
    edge_norm = max([np.abs(graph.edges[e]["weight"]) for e in graph.edges]) / 5
    widths = [np.abs(graph.edges[e]["weight"]) / edge_norm for e in graph.edges]
    edge_colors = define_edge_colors(graph)
    node_colors = define_node_colors(graph)
    node_labels = {
            n: graph.nodes[n]["node_label"]
            if "node_label" in graph.nodes[n].keys()
            else ""
            for n in graph.nodes
    }
    nx.draw(
        graph,
        pos=positions,
        nodelist=graph.nodes,
        with_labels=label_nodes,
        labels=node_labels,
        alpha=0.5,
        node_size=params.NODE_SIZE,
        node_color=node_colors,
        edge_color=edge_colors,
        width=widths,
        connectionstyle="arc3,rad=0.1",
        font_size=params.FONT_SIZE,
        font_color="black",
        ax=ax,
    )
    add_edge_legend(
        ax,
        weights=widths,
        color_list=edge_colors,
        arrow_norm=1 / edge_norm,
        )
    
    if return_pos:
        return ax, positions
    return ax

def position_3d_nodes(x: list, y:list , z:list):
    """
    Distribute the nodes on the 3D axes.
    """
    x_pos = np.linspace(2, 10, len(x))
    y_pos = np.linspace(2, 10, len(y))   
    z_pos = np.linspace(2, 10, len(z))
    nodes_xyz = {node: np.array([x_pos[i], 0,0])
        for i, node in enumerate(x)
        }
    nodes_xyz.update({node: np.array([0, y_pos[i],0])
        for i, node in enumerate(y)
        })
    nodes_xyz.update({node: np.array([0, 0, z_pos[i]])
        for i, node in enumerate(z)
        })
    return nodes_xyz

def plot_xyz(
        graph: nx.DiGraph,
        x: list,
        y: list,
        z: Optional[list] = None,
        sorting: Optional[str | list] = None,
        pos: Optional[dict] = None,
    ):
    """
    Plot the graph in 3D, with the nodes distributed on the x, y and z axes.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to plot.
    x : list
        The nodes to place on the x axis.
    y : list
        The nodes to place on the y axis.
    z : list, optional
        The nodes to place on the z axis, by default None.
    sorting : str or list, optional
        The sorting parameter, by default 'alphabetical'
        Can be a list with [x_sorting, y_sorting, z_sorting]
    pos: dict, optional
        The positions of the nodes, by default None

    Returns
    -------
    plt.Axes
    """

    if sorting is None:
        sorting = 'alphabetical'
    if isinstance(sorting,str):
        sorting = [sorting, sorting, sorting]

    if pos is None:
        if z is None:
            z = [node for node in graph.nodes
                if not node in x and not node in y]
        # sort the nodes
        axis = [x,y,z]
        # sort the nodes
        for index in [0,2,1]: # sort x, then z, then y
            sort_ = sorting[index]
            if sort_ == 'input_clustering':
                refs = [axis[1],axis[2],axis[0]] # will be modified, different from [x,y,z]
                axis[index] = nx_utils.sort_nodes(
                    graph,
                    axis[index],
                    sort_,
                    ref = refs[index]
                    )
            elif sort_ == 'output_clustering':
                refs = [axis[2],axis[0],axis[1]]
                axis[index] = nx_utils.sort_nodes(
                    graph,
                    axis[index],
                    sort_,
                    ref = refs[index]
                    )
            else:
                axis[index] = nx_utils.sort_nodes(graph, axis[index], sort_)

        # distribute the nodes on the axes of the 3d plot
        nodes_xyz = position_3d_nodes(axis[0], axis[1], axis[2])
    else:
        nodes_xyz = pos

    # create the figure
    ax = draw_3d(graph, nodes_xyz)
    return ax

def draw_3d(
        graph: nx.DiGraph,
        nodes_xyz: dict,
):
    """
    Draw a 3D plot of the graph, with positions defined in the nodes_xyz dict.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to plot.
    x : list
        The nodes to place on the x axis.
    y : list
        The nodes to place on the y axis.
    z : list, optional
        The nodes to place on the z axis, by default None.
    sorting : str, optional
        The sorting parameter, by default 'alphabetical'
    pos: dict, optional
        The positions of the nodes, by default None

    Returns
    -------
    plt.Axes
    """
    node_xyz = np.array([nodes_xyz[n] for n in graph.nodes()])
    # edges
    edge_xyz = np.array([(nodes_xyz[u], nodes_xyz[v]) for u,v in graph.edges()])
    # create the figure
    fig = plt.figure(figsize=params.FIGSIZE, dpi=params.DPI)
    ax = fig.add_subplot(111, projection='3d')

    # plot the nodes
    node_colors = define_node_colors(graph)
    ax.scatter(*node_xyz.T, s=20, ec="w", c=node_colors, alpha=0.5)

    # plot the edges
    edge_colors = define_edge_colors(graph)
    for edge_id, edge_ in enumerate(edge_xyz):
        ax.plot(*edge_.T, color=edge_colors[edge_id], alpha=0.1)    

    _format_axes_3d(ax)
    fig.tight_layout()
    return ax

def _format_axes_3d(ax: Axes3D):
        """Visualization options for the 3D axes."""
        # set the camera view
        ax.view_init(elev=10, azim=45)
        # Turn gridlines off
        ax.grid(False)
        # Suppress tick labels
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
            dim.set_ticks([])
        # Set axes labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")