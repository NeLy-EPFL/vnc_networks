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

neuropil = 'LegNp(T3)' # synapses of MDNs in T1/T2/T3
source_neurons = [
    mdn for mdn in mdns if (
        (neuropil in VNC.get_node_label(mdn))
        & ('R' in VNC.get_node_label(mdn))
        )
    ]
leg_motor_neurons = fig1.get_leg_motor_neurons(VNC,leg='h') # uids
target_neurons = VNC.get_neuron_ids({
    'somaSide:string':'RHS',
    })
target_neurons = list(
    set(target_neurons).intersection(leg_motor_neurons)
)
l2_graph = VNC.paths_length_n(2, source_neurons, target_neurons)
sub = VNC.subgraph(l2_graph.nodes, l2_graph.edges())
ax = sub.draw_graph_concentric_by_attribute(
    title='test',
    attribute='target:string',
    center_nodes=source_neurons,
    target_nodes=target_neurons,
    syn_threshold=5,
)
