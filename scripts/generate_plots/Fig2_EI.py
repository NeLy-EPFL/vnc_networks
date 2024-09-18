'''
Plots diving in the excitation-inhibition distributions in the graph.

Fig 2a: Excitation-Inhibition matrix within and across leg neuropils
 (#connections). 

Fig 2b: Excitation-Inhibition matrix within and across leg neuropils
 (summed weights).
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

# -------------------------- Helper functions -------------------------- #
def _nb_local_connections(args):
    # private function for parallelization of 2a
    i, j, neurons_in_neuropils, VNC_matrix = args
    source = list(neurons_in_neuropils.keys())[i]
    target = list(neurons_in_neuropils.keys())[j]
    matrix = copy.deepcopy(VNC_matrix)
    matrix.restrict_from_to(
        row_ids=neurons_in_neuropils[source],
        column_ids=neurons_in_neuropils[target],
        input_type='uid'
        )
    mat = matrix.get_matrix()
    tot = matrix_utils.count_nonzero(mat)
    ex = matrix_utils.count_nonzero(mat, sign='positive')
    inh = matrix_utils.count_nonzero(mat, sign='negative')
    return [tot, ex, inh]

def _sum_local_connections(args):
    # private function for parallelization of 2b
    i, j, neurons_in_neuropils, VNC_matrix = args
    source = list(neurons_in_neuropils.keys())[i]
    target = list(neurons_in_neuropils.keys())[j]
    matrix = copy.deepcopy(VNC_matrix)
    matrix.restrict_from_to(
        row_ids=neurons_in_neuropils[source],
        column_ids=neurons_in_neuropils[target],
        input_type='uid'
        )
    mat = matrix.get_matrix()
    tot = matrix_utils.sum_weights(mat, sign='absolute')
    ex = matrix_utils.sum_weights(mat, sign='positive')
    inh = -1 * matrix_utils.sum_weights(mat, sign='negative')
    return [tot, ex, inh]

# -------------------------- Main functions -------------------------- #

def fig2a():
    # Load all connections
    VNC = all_neurons_helper.get_full_vnc()
    VNC_matrix = VNC.get_cmatrix(type_='unnorm')

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
    nb_entries = len(neurons_in_neuropils.keys())
                
    # For each possible combination of neuropils, compute the number of 
    # excititory and inhibitory connections between them
    total_connections_matrix = np.zeros((nb_entries, nb_entries))
    E_matrix = np.zeros((nb_entries, nb_entries)) # normalised
    I_matrix = np.zeros((nb_entries, nb_entries)) # normalised

    # Parallelize the computation of an independent double for loop
    args_list = []
    result_indices = [
        (i,j)
        for i in range(nb_entries)
        for j in range(nb_entries)
        ]
    for i,j in result_indices:
        args_list.append([i,j,neurons_in_neuropils,VNC_matrix])
    with mp.Pool(mp.cpu_count() - 1) as pool:
        results = pool.map(_nb_local_connections, args_list) # functionalised
    for k, (i, j) in enumerate(result_indices):
            total_connections_matrix[i, j] = results[k][0]
            E_matrix[i, j] = results[k][1]
            I_matrix[i, j] = results[k][2]
                 
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
        ax.set_xticks(range(nb_entries), labels=neurons_in_neuropils.keys())
        ax.set_yticks(range(nb_entries), labels=neurons_in_neuropils.keys())    
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
        cmap=params.grey_heatmap,
        vmin=0,
        vmax=1,
        )
    matrix_design.imshow(
        I_matrix,
        title='Inhibitory connections',
        ax=axs[2],
        cmap=params.grey_heatmap,
        vmin=0,
        vmax=1
        )
    
    # Save the figure
    folder = os.path.join(params.FIG_DIR, 'Fig2')
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, 'Fig2a.pdf'))
    plt.close()

def fig2b():
        # Load all connections
    VNC = all_neurons_helper.get_full_vnc()
    VNC_matrix = VNC.get_cmatrix(type_='unnorm')

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
    nb_entries = len(neurons_in_neuropils.keys())
                
    # For each possible combination of neuropils, compute the number of 
    # excititory and inhibitory connections between them
    total_connections_matrix = np.zeros((nb_entries, nb_entries))
    E_matrix = np.zeros((nb_entries, nb_entries)) # normalised
    I_matrix = np.zeros((nb_entries, nb_entries)) # normalised

    # Parallelize the computation of an independent double for loop
    args_list = []
    result_indices = [
        (i,j)
        for i in range(nb_entries)
        for j in range(nb_entries)
        ]
    for i,j in result_indices:
        args_list.append([i,j,neurons_in_neuropils,VNC_matrix])
    with mp.Pool(mp.cpu_count() - 1) as pool:
        results = pool.map(_sum_local_connections, args_list) # functionalised
    for k, (i, j) in enumerate(result_indices):
            total_connections_matrix[i, j] = results[k][0]
            E_matrix[i, j] = results[k][1]
            I_matrix[i, j] = results[k][2]
                 
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
        ax.set_xticks(range(nb_entries), labels=neurons_in_neuropils.keys())
        ax.set_yticks(range(nb_entries), labels=neurons_in_neuropils.keys())    
    matrix_design.imshow(
        total_connections_matrix,
        title='total connection weights',
        ax=axs[0],
        cmap=params.grey_heatmap,
        vmin=0,
        )
    matrix_design.imshow(
        E_matrix,
        title='Excitatory connection weigths',
        ax=axs[1],
        cmap=params.grey_heatmap,
        vmin=0,
        vmax=1,
        )
    matrix_design.imshow(
        I_matrix,
        title='Inhibitory connection weights',
        ax=axs[2],
        cmap=params.grey_heatmap,
        vmin=0,
        vmax=1
        )
    
    # Save the figure
    folder = os.path.join(params.FIG_DIR, 'Fig2')
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, 'Fig2b.pdf'))
    plt.close()

if __name__ == '__main__':
    #fig2a()
    #fig2b()
    pass