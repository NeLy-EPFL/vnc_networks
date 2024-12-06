'''
Plots diving in the excitation-inhibition distributions in the graph.

Fig 2a: Excitation-Inhibition matrix within and across leg neuropils
 (#connections). 

Fig 2b: Excitation-Inhibition matrix within and across leg neuropils
 (summed weights).

Fig 2c: Cumulated E and I weights up to n hops (4) from source neuron to target
neuron, based on normalised matrix multiplication. 
Specific use case: MDN|Ti to MNs|Ti, get a quantification of how much 
innervation there is and whether T1 is more linear than T3.
'''
import typing
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import multiprocessing as mp

import specific_neurons.all_neurons_helper as all_neurons_helper
import specific_neurons.motor_neurons_helper as mns_helper
import specific_neurons.mdn_helper as mdn_helper
import utils.matrix_utils as matrix_utils
import utils.matrix_design as matrix_design
import params

FOLDER_NAME = 'explo_exc-inhib'
FOLDER = os.path.join(params.FIG_DIR, FOLDER_NAME)
os.makedirs(FOLDER, exist_ok=True)

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
def simple_count_in_neuropil(
    side: typing.Literal['L', 'R'],
    leg: typing.Literal['f', 'm', 'h'], 
):
    """
    Count the number of leg motor neurons reached from MDN by restricting
    neurons to a neuropil.
    """
    if side not in ['L','R']:
        raise ValueError('side must be either L or R')
    if leg not in ['f', 'm', 'h']:
        raise ValueError('leg must be either f, m or h')
    match leg:
        case 'f':
            neuropil = 'T1'
        case 'm':
            neuropil = 'T2'
        case 'h':
            neuropil = 'T3'   
    # Load all connections
    VNC = all_neurons_helper.get_full_vnc()
    # Get relevant neurons
    neurons_used = []
    neurons_used.extend( # intereurons
        VNC.get_neurons_in_neuropil(
            neuropil=neuropil, side=side+'HS'
            )
        )
    target = list(mns_helper.get_leg_motor_neurons(
            data=VNC, leg=leg, side=side+'HS'
            )) # motor neurons
    print(f"total number of {leg}{side} leg motor neurons: {len(target)}")
    neurons_used.extend(target)
    source = mdn_helper.get_mdn_uids(VNC) # MDN
    neurons_used.extend(source)
    neurons_used = list(set(neurons_used))

    # Restrict the matrix and compute second order connections
    cm = VNC.get_cmatrix(type_='syn_count')
    cm.restrict_nodes(neurons_used)
    cm.power_n(2)

    # Get number of connections from source to target
    cm.restrict_from_to(source, target, input_type='uid')
    list_down = cm.list_downstream_neurons(source)
    print(f'Number of {leg} motor neurons reached from MDN in {neuropil} {side}: {len(list_down)}')

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
    plt.savefig(os.path.join(FOLDER, 'Fig2a.pdf'))
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
    plt.savefig(os.path.join(FOLDER, 'Fig2b.pdf'))
    plt.close()

def fig2c():
    pass

if __name__ == '__main__':
    simple_count_in_neuropil('R', 'h')
    simple_count_in_neuropil('L', 'h')
    simple_count_in_neuropil('R', 'f')
    simple_count_in_neuropil('L', 'f')
    #fig2a()
    #fig2b()
    #fig2c()
    pass