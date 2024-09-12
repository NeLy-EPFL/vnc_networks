'''
Plots diving in the excitation-inhibition distributions in the graph.

Fig 2a: Excitation-Inhibition matrix within and across neuropils. 
'''
import numpy as np

import specific_neurons.all_neurons_helper as all_neurons_helper

def fig2a():

    # Load all connections
    VNC = all_neurons_helper.get_full_vnc()

    # List of neurons in relevant neuropils structured as a dictionary
    # {neuropil: [neuron_ids]}
    # with neuropil in ['T1_L', 'T1_R', 'T2_L', 'T2_R', 'T3_L', 'T3_R']
    neurons_in_neuropils = {}
    for neuropil in ['T1', 'T2', 'T3']:
        for side in ['L', 'R']:
            neuropil_ = f'{neuropil}_{side}'
            neurons_in_neuropils[neuropil_] = VNC.get_neurons_in_neuropil(
                neuropil=neuropil, side=side
                )
    
    # For each possible combination of neuropils, compute the number of 
    # excititory and inhibitory connections between them
    total_connections_matrix = np.zeros((6, 6))
    E_matrix = np.zeros((6, 6)) # normalised by total connections
    I_matrix = np.zeros((6, 6)) # normalised by total connections

    for i, source in enumerate(neurons_in_neuropils.keys()):
        for j, target in enumerate(neurons_in_neuropils.keys()):
            connections = VNC.get_connections_from_to(
                source=neurons_in_neuropils[source], 
                target=neurons_in_neuropils[target]
                )
            total_connections_matrix[i, j] = len(connections)
            E_matrix[i, j] = np.sum(connections['type'] == 'excitatory')
            I_matrix[i, j] = np.sum(connections['type'] == 'inhibitory')



if __name__ == '__main__':
    fig2a()
    pass