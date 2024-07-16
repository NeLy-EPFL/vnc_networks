"""
author: femke.hurtak@epfl.ch

Script to load the MANC dataset and save it in a usable format.
"""
from neuron import Neuron
from connections import Connections
from get_nodes_data import get_neuron_bodyids


neurons_pre = get_neuron_bodyids({'type:string': 'MDN'})

VNC = Connections(from_file='VNC_split_MDN')  # full VNC
connections = VNC.get_connections()
MDN_outputs = connections[
    (connections[':START_ID(Body-ID)'].isin(neurons_pre))
    ]
print(MDN_outputs)


# TODO
# - color as a function of the neuropil
