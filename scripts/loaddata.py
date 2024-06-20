from get_nodes_data import get_neurons_from_class, get_neuron_ids, load_data_neuron
from connections import Connections

import utils.matrix_utils as matrix_utils

# Load neurons
neurons_pre = get_neuron_ids({'type:string': 'MDN'})
neuropil = 'T1' 
neurons_post = get_neurons_from_class('motor neuron')

# Load dataset
connections = Connections() # entire dataset

# Compute second order connections
neurons = list(set(neurons_pre).union(set(neurons_post)))
cmatrix = connections.get_cmatrix()
cmatrix.power_n(2)
cmatrix.restrict_nodes(neurons)
