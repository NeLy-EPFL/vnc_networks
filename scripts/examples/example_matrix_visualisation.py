from connections import Connections
from connectome_reader import ConnectomeReader

# Load dataset
CR = ConnectomeReader("MANC", "v1.0")
# Load neurons
neurons_pre = CR.get_neuron_bodyids({'type': 'MDN'})
neuropil = 'T1' 
#neurons_post = get_neurons_from_class('motor neuron')
neurons_post = CR.get_neuron_bodyids(
    {'neuropil': neuropil, 'class_1': 'motor'}
)
# Load dataset
connections = Connections(CR=CR) # entire dataset

# Compute second order connections
cmatrix = connections.get_cmatrix(type_='norm')
cmatrix.power_n(2)
cmatrix.restrict_from_to(neurons_pre, neurons_post,  input_type='body_id')
cmatrix.cluster_hierarchical_unilateral(axis='column')
cmatrix.imshow(title='second_order_connections_MDN_MNs')
#cmatrix.spy(title='second_order_connections_MDN_MNs')
