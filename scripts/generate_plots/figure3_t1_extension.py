"""
Figure 3: T1 extension

Show that exciting MDN favors leg extension in T1.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import copy

import params
import specific_neurons.mdn_helper as mdn_helper
import specific_neurons.motor_neurons_helper as mns_helper
from utils import matrix_design



FOLDER_NAME = 'Figure_3_t1_extension'
FOLDER = os.path.join(params.FIG_DIR, FOLDER_NAME)
os.makedirs(FOLDER, exist_ok=True)

# -------------------------- Helper functions -------------------------- #

# -------------------------- Main functions -------------------------- #
def confusion_matrix_mdn_to_mn(n_hops: int = 2):
    '''
    Create a confusion matrix of the number of connections from MDN to motor
    neurons within n hops, where the rows are the MDNs split by neuropil and
    the columns are the motor neurons split by leg.
    '''
    # Loading the connectivity data
    full_VNC = mdn_helper.get_vnc_split_MDNs_by_neuropil(
        not_connected=mdn_helper.get_mdn_bodyids()
        ) # exclude connections from MDNs to MDNs
    VNC = full_VNC.get_connections_with_only_traced_neurons() # exclude untraced neurons for statistics
    
    # Get the uids of neurons split by MDN synapses in leg neuropils
    mdn_uids = VNC.get_neuron_ids({'type:string': 'MDN'})
    mdn_neuropil = []
    for i in range(3):
        neuropil = 'LegNp(T'+str(i+1)+')'
        mdn_neuropil.append([
            uid for uid in mdn_uids if neuropil in VNC.get_node_label(uid)
            ]) # 2 left and 2 right MDNs
        
    # Get the uids of motor neurons split by leg
    list_motor_neurons = [[],[],[]]
    for i, leg in enumerate(['f', 'm', 'h']):
        leg_motor_neurons = list(mns_helper.get_leg_motor_neurons(VNC, leg=leg))
        list_motor_neurons[i] = leg_motor_neurons

    # Get the summed connection strength up to n hops
    eff_weight_abs = VNC.get_cmatrix(type_='norm')
    eff_weight_abs.absolute()
    eff_weight_abs.within_power_n(n_hops)

    # Get the confusion matrix
    confusion_matrix = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            mat = copy.deepcopy(eff_weight_abs)
            mat.restrict_from_to(
                row_ids=mdn_neuropil[i],
                column_ids=list_motor_neurons[j],
                input_type='uid'
                )
            matrix = mat.get_matrix()
            confusion_matrix[i,j] = matrix.sum()

    # Plot the confusion matrix
    ax = matrix_design.imshow(
        confusion_matrix,
        ylabel='MDN subdivision',
        row_labels=['T1', 'T2', 'T3'],
        xlabel='leg motor neuron',
        col_labels=['f', 'm', 'h'],
        title='Confusion matrix of MDN to MN connections',
        cmap=params.grey_heatmap,
        vmin=0,
        )
    plt.savefig(os.path.join(FOLDER, f'Fig3_{n_hops}_hops_total_weight.pdf'))
    plt.close()



if __name__ == '__main__':
    confusion_matrix_mdn_to_mn(n_hops=4)
    pass