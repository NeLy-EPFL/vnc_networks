from get_nodes_data import get_neurons_from_class, get_neuron_bodyids
from connections import Connections

neuropil = 'T1'
neurons = get_neurons_from_class('descending neuron')
# Load dataset
connections = Connections(neurons_pre=neurons) # entire dataset
connections.initialize()

# Compute second order connections
cmatrix = connections.get_cmatrix(type_='norm')
lookup2 = cmatrix.get_lookup()
cmatrix.spy(title='test_before')

#
subneurons = lookup2['body_id'].values[0:500]
cmatrix.restrict_from_to(subneurons, subneurons,  input_type='body_id')

#cmatrix.imshow(title='test2')
cmatrix.spy(title='test2')
