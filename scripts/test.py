from neuron import Neuron
from connections import Connections

MDN0 = Neuron(from_file='MDN0')
MDN1 = Neuron(from_file='MDN1')


VNC = Connections(from_file='full_VNC')  # split VNC
#VNC = Connections()
#VNC.initialize(split_neurons=[MDN0, MDN1])
connections = VNC.get_connections()
print(connections[
    (connections[':END_ID(Body-ID)'] == 10075)
    & (connections[':START_ID(Body-ID)'] == 14523)
    ])