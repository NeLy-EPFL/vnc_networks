"""
author: femke.hurtak@epfl.ch

Script to load the MANC dataset and save it in a usable format.
"""

import matplotlib.pyplot as plt
import pandas as pd
from connections import Connections
from connectome_reader import ConnectomeReader

CR = ConnectomeReader("MANC", "v1.0")
# MDNs = get_neuron_ids({'type:string': 'MDN'})
DNxn050 = CR.get_neuron_bodyids({"name": "DNxn050"})  # MDN R
DNxn049 = CR.get_neuron_bodyids({"name": "DNxn049"})  # MDN L

# list(set(t1).union(neurons_pre))
connections = Connections(CR=CR)  # entire dataset
# merge nodes that are very similar
connections.merge_nodes(connections.get_uids_from_bodyids(DNxn050))
connections.merge_nodes(connections.get_uids_from_bodyids(DNxn049))

neurons_pre = connections.get_neuron_ids({"type": "MDN"})

for neuropil in ["T1", "T2", "T3"]:
    neurons_post = connections.get_neuron_ids(
        {"neuropil": neuropil, "class_1": "motor"}
    )

    subconnections = connections.subgraph_from_paths(
        neurons_pre, neurons_post, n_hops=2
    )
    # plots

    subconnections.display_graph_per_attribute(
        attribute="neuropil",
        center=neurons_pre,
        title=f"MDN-merged_to_{neuropil}_MNs_2_hops_path_classes_neuromere",
    )
    subconnections.display_graph_per_attribute(
        attribute="exit_nerve",
        center=neurons_pre,
        title=f"MDN-merged_to_{neuropil}_MNs_2_hops_path_exitnerve",
    )
    subconnections.draw_3d_custom_axis(
        x=neurons_pre,
        y=neurons_post,
        y_sorting="input_clustering",
        title=f"MDN-merged_to_{neuropil}_MNs_2_hops_path_z-input_y-input",
    )
    del subconnections, neurons_post


# nx_design.draw_3d(t1_l2_graph,x=neurons_pre,y=t1_neurons_post,sorting='exitNerve:string')
