'''
Figure 5: T3 Oscillations

Show how oscillations and coordinated movements are generated in T3.
'''

import os
import matplotlib.pyplot as plt

import params
import mdn_paper_helper_functions as paper_funcs


FOLDER_NAME = 'Figure_5_t3_oscillations'
FOLDER = os.path.join(params.FIG_DIR, FOLDER_NAME)
os.makedirs(FOLDER, exist_ok=True)

# -------------------------- Helper functions -------------------------- #


# -------------------------- Main functions -------------------------- #

def hind_leg_muscles_graph(
        muscle_: str,
        side_: str = 'RHS',
        n_hops: int = 2,
        label_nodes: bool = True,
        ):
    '''
    Show the graph of the neurons contacting MDNs -> motor neurons within n hops
    for the front leg muscles.
    '''
    target = {
        'class:string': 'motor neuron',
        'somaSide:string': side_,
        'subclass:string': 'hl',
        'target:string': muscle_,
        }
    title, axs = paper_funcs.graph_from_mdn_to_muscle(
        target,
        n_hops=n_hops,
        label_nodes=label_nodes,
        )
    plt.savefig(os.path.join(FOLDER, title+'.pdf'))
    plt.close()


if __name__ == '__main__':
    #hind_leg_muscles_graph(muscle_ = 'Tr flexor')
    pass