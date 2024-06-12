"""
author: femke.hurtak@epfl.ch

Script to load the MANC dataset and save it in a usable format.
"""

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

import params
from get_nodes_data import get_neurons_from_class, get_neuron_ids, load_data_neuron
from connections import Connections

import utils.nx_design as nx_design

# Load the nodes and connections data


# Load the neuprint data
#connections = pd.read_feather(params.NEUPRINT_CONNECTIONS_FILE)
#print(connections.head())

#print(connections[['roiInfo:string']].iloc[3225307].values[0])
#print(connections.columns)
# >Index([':START_ID(Body-ID)', 'weightHR:int', 'weight:int', 'weightHP:int',
# >       ':END_ID(Body-ID)', 'roiInfo:string'],
# >      dtype='object')
# Connection file: from where (id) to where (id), with a dictionary of where the synapses are 


#neurons = pd.read_feather(params.NEUPRINT_NODES_FILE)
#top = neurons.head()
#for c in top.columns:
#    print(c)
#    print(top[c])
#    print('---')
# Example use cases
#ids = get_neurons_from_class('descending neuron')
#MDNs = get_neuron_ids({'type:string': 'MDN'})
#MDN_info = load_data_neuron(MDNs[0])
#print(MDN_info)

'''
neurons_pre = get_neurons_from_class('descending neuron')
#neurons_pre = get_neuron_ids({'type:string': 'MDN'})
neurons_post = get_neurons_from_class('motor neuron')
#neurons = list(set(neurons_pre).union(set(neurons_post)))
connections = Connections(neurons_pre, neurons_post)
#connections.reorder_neurons(by = 'list', order = ordered_neurons)
#connections.display_adjacency_matrix(title = 'DNs_to_MNs', method = "heatmap")
#connections.display_graph(title = 'DNs_to_MNs')
#connections.display_graph_per_attribute(
#    attribute = 'exitNerve:string',
#    center = 'None',
#    title = 'DNs_to_MNs_per_neuropil',
#    )
'''


'''
connections = Connections() # entire dataset
neurons_pre = get_neuron_ids({'type:string': 'MDN'})
neurons_post = get_neurons_from_class('motor neuron')
neurons = list(set(neurons_pre).union(set(neurons_post)))
graph_2 = connections.get_nx_graph(hops=2, nodes=neurons)
#attribute = 'exitNerve:string'
ax = nx_design.draw_graph_concentric_circles(graph_2)
plt.savefig('MDN_and_MNs_2_hops.png')
'''

t1 = get_neuron_ids(
    {'somaNeuromere:string': 'T1'}
    )
neurons_pre = get_neuron_ids({'type:string': 'MDN'})
#list(set(t1).union(neurons_pre))
connections = Connections() # entire dataset
connections.initialize() # initialize the graph
neurons_post = get_neuron_ids(
    {'somaNeuromere:string': 'T1', 'class:string': 'motor neuron'}
    )
l2_graph = connections.paths_length_n(2, neurons_pre, neurons_post)
#t2_l2_graph = connections.paths_length_n(2, neurons_pre, t2_neurons_post)
#t3_l2_graph = connections.paths_length_n(2, neurons_pre, t3_neurons_post)
subconnections = connections.subgraph(l2_graph.nodes) # new Connections object
#subconnections.display_graph_per_attribute(
#    attribute = 'somaNeuromere:string',#'somaNeuromere:string', #'exitNerve:string',
#    center = neurons_pre,
#    title = 'MDN_to_T3_MNs_2_hops_path_classes_neuromere',
#    )
#subconnections.display_graph_per_attribute(
#    attribute = 'exitNerve:string',#'somaNeuromere:string', #'exitNerve:string',
#    center = neurons_pre,
#    title = 'MDN_to_T3_MNs_2_hops_path_exitnerve',
#)


subconnections.draw_3d_custom_axis(
    x=neurons_pre,
    y=neurons_post,
    title='MDN_to_T1_MNs_2_hops_path_input_clustering'
    )

#nx_design.draw_3d(t1_l2_graph,x=neurons_pre,y=t1_neurons_post,sorting='exitNerve:string')



