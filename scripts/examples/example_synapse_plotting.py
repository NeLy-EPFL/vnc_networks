"""
author: femke.hurtak@epfl.ch

Script to load the MANC dataset and save it in a usable format.
"""

import matplotlib.pyplot as plt
import pandas as pd

from vnc_networks import params
from vnc_networks.connectome_reader import MANC
from vnc_networks.neuron import Neuron
from vnc_networks.utils import plots_design

CR = MANC('v1.0')
synapses = []
neurons_pre = CR.get_neuron_bodyids({"type": "MDN"})

for i in range(4):
    MDN = Neuron(neurons_pre[i], CR=CR)
    # MDN = Neuron(from_file='MDN'+str(i))
    syn = MDN.get_synapse_distribution()
    synapses.append(syn)
    MDN.cluster_synapses_spatially(n_clusters=5)
    MDN.create_synapse_groups(attribute="KMeans_cluster")
    MDN.save(name="MDN" + str(i))
    MDN.plot_synapse_distribution(color_by="KMeans_cluster")


"""
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

# Update and make other plots
MDN = Neuron(from_file='MDN0')
synapses = MDN.get_synapse_table()
_ = MDN.get_synapse_distribution()
MDN.cluster_synapses_spatially(n_clusters=5)
MDN.plot_synapse_distribution(color_by='KMeans_cluster')
MDN.plot_synapse_distribution(color_by=None)
MDN.create_synapse_groups(attribute='KMeans_cluster')
MDN.save(name='MDN0')
"""
