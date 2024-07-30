"""
author: femke.hurtak@epfl.ch

Script to load the MANC dataset and save it in a usable format.
"""
from neuron import Neuron
from connections import Connections
from get_nodes_data import get_neuron_bodyids
import params

from matplotlib_venn import venn3
import matplotlib.pyplot as plt
import seaborn as sns

MDNs = []
neurons_pre = get_neuron_bodyids({'type:string': 'MDN'})

#for i in range(4):
    #MDN = Neuron(neurons_pre[i])
    #_ = MDN.get_synapse_distribution(threshold=True)
    #MDN = Neuron(from_file='MDN_split-neuropil_'+str(i))  # if already defined
    #MDN.cluster_synapses_spatially(n_clusters=3)
    #MDN.create_synapse_groups(attribute='KMeans_cluster')
    #MDN.create_synapse_groups(attribute='neuropil')
    #MDN.plot_synapse_distribution(
    #    color_by='neuropil',
    #    discrete_coloring=True,
    #    threshold=True,
    #    cmap="Spectral")
    #MDN.save(name='MDN_split-neuropil_'+str(i))  # if you want to save the neuron
    #MDNs.append(MDN)



#VNC = Connections()  # full VNC
#VNC.initialize(split_neurons=MDNs)  # split MDNs according to the synapse data
#VNC.save(name='VNC_split_MDNs_by_neuropil')  # if you want to reuse it later
#connections = VNC.get_connections()


VNC = Connections(from_file='VNC_split_MDNs_by_neuropil')  # load the split VNC
MDNs = VNC.subgraph(nodes=neurons_pre)
MDNs.display_graph(label_nodes=True)

'''
# TODO
# - color as a function of the neuropil

fig,ax = plt.subplots(nrows = 1, ncols = 4, figsize = (20,5))

# Plot the Venn diagram of the synapse groups
for i in range(4):
    MDN0_down = table[table[':START_ID(Body-ID)'] == neurons_pre[i]]

    down_sub_0 = MDN0_down[
        MDN0_down['subdivision_start'] == 0
        ]
    down_0 = set(down_sub_0[':END_ID(Body-ID)'].values)
    n_syn_0 = down_sub_0['syn_count'].sum()

    down_sub_1 = MDN0_down[
        MDN0_down['subdivision_start'] == 1
        ]
    down_1 = set(down_sub_1[':END_ID(Body-ID)'].values)
    n_syn_1 = down_sub_1['syn_count'].sum()

    down_sub_2 = MDN0_down[
        MDN0_down['subdivision_start'] == 2
        ]
    down_2 = set(down_sub_2[':END_ID(Body-ID)'].values)
    n_syn_2 = down_sub_2['syn_count'].sum()


    #colors = sns.color_palette(palette = 'crest', n_colors = 3)
    colors = params.custom_palette[:3]
    v = venn3(
        [down_0, down_1, down_2],
        (f'0: {n_syn_0}', f'1: {n_syn_1}', f'2: {n_syn_2}'),
        ax=ax[i],
        set_colors = colors,
        alpha = 0.8
        )
    ax[i].set_title(f'MDN{i}')


#plt.title('MDN0_downstream_by_branches')
plt.savefig('MDN0_downstream_by_branches.pdf')

'''