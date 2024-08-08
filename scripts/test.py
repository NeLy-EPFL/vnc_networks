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
print(VNC.list_possible_attributes())