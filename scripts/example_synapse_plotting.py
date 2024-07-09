"""
author: femke.hurtak@epfl.ch

Script to load the MANC dataset and save it in a usable format.
"""
import pandas as pd
import matplotlib.pyplot as plt

from get_nodes_data import get_neuron_bodyids
from neuron import Neuron
import utils.plots_design as plots_design

import params

synapses = []
for i in range(4):
    neurons_pre = get_neuron_bodyids({'type:string': 'MDN'})
    MDN = Neuron(neurons_pre[i])
    MDN.get_synapse_distribution('post')
    MDN.save(name='MDN'+str(i))
    MDN = Neuron(from_file = 'MDN'+str(i))
    MDN.plot_synapse_distribution('post')
    syn = MDN.get_synapse_distribution('post')
    synapses.append(syn)

fig, ax = plt.subplots(1, 1, figsize=params.FIGSIZE, dpi=params.DPI)
for syn in synapses:
    ax.scatter(syn[0], syn[1], c=syn[2], cmap=params.red_heatmap, marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
cbar = plt.colorbar(
    ax.collections[0],
    ax=ax,
    orientation='vertical',
    label='Z',
    )
cbar = plots_design.make_nice_cbar(cbar)
ax = plots_design.make_nice_spines(ax)
plt.tight_layout()
plt.savefig(f"{params.PLOT_DIR}/synapse_distribution_post_MDNs.pdf")

