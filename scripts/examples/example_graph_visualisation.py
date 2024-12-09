"""
author: femke.hurtak@epfl.ch

Script to load the MANC dataset and save it in a usable format.
"""

import matplotlib.pyplot as plt
import pandas as pd
from connections import Connections
from get_nodes_data import get_neuron_bodyids, get_neurons_from_class, load_data_neuron

# MDNs = get_neuron_ids({'type:string': 'MDN'})
DNxn050 = get_neuron_bodyids({"systematicType:string": "DNxn050"})  # MDN R
DNxn049 = get_neuron_bodyids({"systematicType:string": "DNxn049"})  # MDN L

# list(set(t1).union(neurons_pre))
connections = Connections()  # entire dataset
connections.initialize()  # initialize the graph
# merge nodes that are very similar
connections.merge_nodes(connections.get_uids_from_bodyids(DNxn050))
connections.merge_nodes(connections.get_uids_from_bodyids(DNxn049))

neurons_pre = connections.get_neuron_ids({"type:string": "MDN"})

for neuropil in ["T1", "T2", "T3"]:
    neurons_post = connections.get_neuron_ids(
        {"somaNeuromere:string": neuropil, "class:string": "motor neuron"}
    )
    l2_graph = connections.paths_length_n(2, neurons_pre, neurons_post)
    subconnections = connections.subgraph(
        l2_graph.nodes,
        edges=l2_graph.edges(),
    )  # new Connections object
    # plots

    subconnections.display_graph_per_attribute(
        attribute="somaNeuromere:string",  #'somaNeuromere:string', #'exitNerve:string',
        center=neurons_pre,
        title=f"MDN-merged_to_{neuropil}_MNs_2_hops_path_classes_neuromere",
    )
    subconnections.display_graph_per_attribute(
        attribute="exitNerve:string",  #'somaNeuromere:string', #'exitNerve:string',
        center=neurons_pre,
        title=f"MDN-merged_to_{neuropil}_MNs_2_hops_path_exitnerve",
    )
    subconnections.draw_3d_custom_axis(
        x=neurons_pre,
        y=neurons_post,
        y_sorting="input_clustering",
        title=f"MDN-merged_to_{neuropil}_MNs_2_hops_path_z-input_y-input",
    )
    del subconnections, neurons_post, l2_graph


# nx_design.draw_3d(t1_l2_graph,x=neurons_pre,y=t1_neurons_post,sorting='exitNerve:string')
