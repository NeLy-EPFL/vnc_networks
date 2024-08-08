import pandas as pd
import os
import params
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import networkx as nx

from neuron import Neuron
from connections import Connections
import generate_plots.Fig1_connections as fig1

VNC = fig1.get_vnc_split_MDNs_by_neuropil(not_connected=fig1.get_mdn_bodyids())

mdns = VNC.get_neuron_ids({
        #'somaSide:string': 'RHS',
        'type:string': 'MDN'
        })

neuropil = 'LegNp(T1)' # synapses of MDNs in T1/T2/T3
source_neurons = [
    mdn for mdn in mdns if (
        (neuropil in VNC.get_node_label(mdn))
        & ('R' in VNC.get_node_label(mdn))
        )
    ]
target_neurons = VNC.get_neuron_ids({
    'somaNeuromere:string': f'T1',
    'somaSide:string': 'RHS',
    'class:string': 'motor neuron'
    })
l2_graph = VNC.paths_length_n(2, source_neurons, target_neurons)
sub = VNC.subgraph(l2_graph.nodes, l2_graph.edges())
ax = sub.display_graph(
    title='test'
)
plt.savefig('test.png')