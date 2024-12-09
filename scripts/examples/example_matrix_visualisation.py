from connections import Connections
from get_nodes_data import get_neuron_bodyids, get_neurons_from_class

# Load neurons
neurons_pre = get_neuron_bodyids({'type:string': 'MDN'})
neuropil = 'T1' 
#neurons_post = get_neurons_from_class('motor neuron')
neurons_post = get_neuron_bodyids(
    {'somaNeuromere:string': neuropil, 'class:string': 'motor neuron'}
)
# Load dataset
connections = Connections() # entire dataset
connections.initialize()

# Compute second order connections
cmatrix = connections.get_cmatrix(type_='norm')
cmatrix.power_n(2)
cmatrix.restrict_from_to(neurons_pre, neurons_post,  input_type='body_id')
cmatrix.cluster_hierarchical_unilateral(axis='column')
cmatrix.imshow(title='second_order_connections_MDN_MNs')
#cmatrix.spy(title='second_order_connections_MDN_MNs')
