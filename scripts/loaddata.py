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


MDN = Neuron(from_file='MDN0')
_ = MDN.get_synapse_distribution('post')
MDN.cluster_synapses_spatially(pre_or_post=None, n_clusters=3)
MDN.plot_synapse_distribution(pre_or_post=None, color_by='KMeans_cluster')
MDN.plot_synapse_distribution(pre_or_post=None, color_by=None)


# TODO
# - color as a function of the neuropil
# - implement spatial clustering
# - interface the synapse data with the larger connectivity graph to see specific patterns
