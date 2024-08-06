import pandas as pd
import os
import params
import numpy as np
import scipy as sc

from neuron import Neuron
from connections import Connections

VNC = Connections(from_file='VNC_split_MDNs_by_neuropil')
neuropil = 'T1'
selection_dict = {
    #'somaNeuromere:string': neuropil,
    'class:string': 'motor neuron',
    'subclass:string': 'fl'
    }
neurons_post = VNC.get_neuron_ids(selection_dict)
print(neurons_post)
print(len(neurons_post))
labels = [VNC.get_node_label(uid) for uid in neurons_post]
print(labels)
print(len(labels))
print(len(set(labels)))