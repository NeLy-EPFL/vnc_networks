'''
Plots diving in the excitation-inhibition distributions in the graph.

Fig 2a: Excitation-Inhibition matrix within and across neuropils. 
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import multiprocessing as mp


import specific_neurons.all_neurons_helper as all_neurons_helper
import utils.matrix_utils as matrix_utils
import utils.matrix_design as matrix_design
import params

def fig2a():

    # Load all connections
    VNC = all_neurons_helper.get_full_vnc()
    VNC_matrix = VNC.get_cmatrix(type_='syn_count')

    # List of neurons in relevant neuropils structured as a dictionary
    # {neuropil: [neuron_ids]}
    # with neuropil in ['T1_L', 'T1_R', 'T2_L', 'T2_R', 'T3_L', 'T3_R']
    neurons_in_neuropils = {}
    for neuropil in ['T1', 'T2', 'T3']:
        for side in ['LHS', 'RHS']:
            neuropil_ = f'{neuropil}_{side}'
            neurons_in_neuropils[neuropil_] = VNC.get_neurons_in_neuropil(
                neuropil=neuropil, side=side
                )
                
    # For each possible combination of neuropils, compute the number of 
    # excititory and inhibitory connections between them
    total_connections_matrix = np.zeros((6, 6))
    E_matrix = np.zeros((6, 6)) # normalised by total connections
    I_matrix = np.zeros((6, 6)) # normalised by total connections

    def nb_local_connections(tuple_ij):
        i = tuple_ij[0]
        j = tuple_ij[1]
        source = list(neurons_in_neuropils.keys())[i]
        target = list(neurons_in_neuropils.keys())[j]
        matrix = copy.deepcopy(VNC_matrix)
        matrix.restrict_from_to(
            row_ids=neurons_in_neuropils[source],
            column_ids=neurons_in_neuropils[target],
            input_type='uid'
            )
        mat = matrix.get_matrix()
        total_connections_matrix[i, j] = matrix_utils.count_nonzero(mat)
        E_matrix[i, j] = matrix_utils.count_nonzero(mat, sign='positive')
        I_matrix[i, j] = matrix_utils.count_nonzero(mat, sign='negative')

    # Parallelize the computation
    with mp.Pool() as pool:
        pool.map(
            nb_local_connections,
            [(i,j) for i in range(6) for j in range(6)]
            )
                 
    # normalise by total connections
    E_matrix = E_matrix / total_connections_matrix
    I_matrix = I_matrix / total_connections_matrix

    _, axs = plt.subplots(
        1,
        3,
        figsize=(3 * params.FIG_WIDTH, params.FIG_HEIGHT),
        dpi=params.DPI,
        )
    for ax in axs:
        ax.set_xticks(range(1,7), labels=neurons_in_neuropils.keys())
        ax.set_yticks(range(1,7), labels=neurons_in_neuropils.keys())    
    matrix_design.imshow(
        total_connections_matrix,
        title='total connections',
        ax=axs[0],
        cmap=params.grey_heatmap,
        vmin=0,
        )
    matrix_design.imshow(
        E_matrix,
        title='Excitatory connections',
        ax=axs[1],
        cmap=params.red_heatmap,
        vmin=0,
        )
    matrix_design.imshow(
        I_matrix,
        title='Inhibitory connections',
        ax=axs[2],
        cmap=params.blue_heatmap,
        vmin=0,
        )
    
    # Save the figure
    folder = os.path.join(params.FIG_DIR, 'Fig2')
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, 'Fig2a.pdf'))
    plt.close()



if __name__ == '__main__':
    fig2a()
    pass