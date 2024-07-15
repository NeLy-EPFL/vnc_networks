"""
author: femke.hurtak@epfl.ch

Script to load the MANC dataset and save it in a usable format.
"""
from neuron import Neuron
from connections import Connections
from get_nodes_data import get_neuron_bodyids

MDNs = []
neurons_pre = get_neuron_bodyids({'type:string': 'MDN'})

for i in range(4):
    MDN = Neuron(neurons_pre[i])
    #MDN = Neuron(from_file='MDN'+str(i))  # if already defined
    MDN.cluster_synapses_spatially(n_clusters=5)
    MDN.create_synapse_groups(attribute='KMeans_cluster')
    #MDN.save(name='MDN'+str(i))  # if you want to save the neuron
    MDNs.append(MDN)

VNC = Connections()  # full VNC
VNC.initialize(split_neurons=MDNs)  # split MDNs according to the synapse data
VNC.save(name='VNC_split_MDN')  # if you want to reuse it later
connections = VNC.get_connections()
print(connections.head())

# TODO
# - color as a function of the neuropil
