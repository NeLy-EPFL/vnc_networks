from get_nodes_data import get_neurons_from_class, get_neuron_ids
from connections import Connections

# Load neurons
neurons_pre = get_neuron_ids({'type:string': 'MDN'})
neuropil = 'T1' 
neurons_post = get_neurons_from_class('motor neuron')

# Load dataset
neurons = list(set(neurons_pre).union(set(neurons_post)))
connections = Connections(neurons) # entire dataset
connections.initialize()

# Compute second order connections
neurons = list(set(neurons_pre).union(set(neurons_post)))
cmatrix = connections.get_cmatrix()
cmatrix.power_n(2)
cmatrix.restrict_nodes(neurons)
cmatrix.spy(title='second_order_connections_MDN_MNs')
