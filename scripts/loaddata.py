"""
author: femke.hurtak@epfl.ch

Script to load the MANC dataset and save it in a usable format.
"""

import pandas as pd
import matplotlib.pyplot as plt

import params
from get_nodes_data import get_neurons_from_class, get_neuron_ids, load_data_neuron
from connections import Connections

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


#neurons_pre = get_neurons_from_class('descending neuron')
neurons_pre = get_neuron_ids({'type:string': 'MDN'})
neurons_post = get_neurons_from_class('motor neuron')
#neurons = list(set(neurons_pre).union(set(neurons_post)))
connections = Connections(neurons_pre, neurons_post)
connections.initialize()
#connections.reorder_neurons(by = 'list', order = ordered_neurons)
connections.display_adjacency_matrix(title = 'MDNs_to_MNs', method = "heatmap")