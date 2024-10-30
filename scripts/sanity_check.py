from cmatrix import CMatrix
from connections import Connections
import specific_neurons.all_neurons_helper as all_neurons
import specific_neurons.mdn_helper as mdn_helper
import params


import numpy as np
import pandas as pd

# MDN bodyids:
mdn_bodyids = mdn_helper.get_mdn_bodyids()
mdn_0 = mdn_bodyids[0]
print('Selected MDN: ', mdn_0)


# Our connections object:
full_VNC = all_neurons.get_full_vnc()
VNC = full_VNC.get_connections_with_only_traced_neurons()
down_VNC_graph = VNC.get_neurons_downstream_of(
    neuron_id = mdn_0,
    input_type='body_id',
    output_type='body_id',
    )

print(f'Downstream VNC graph (total {len(down_VNC_graph)})')
uid_mdn_0 = full_VNC.get_uids_from_bodyid(mdn_0)[0]
uids_down_VNC_graph = [
    full_VNC.get_uids_from_bodyid(bodyid)[0] for bodyid in down_VNC_graph
    ]
edges_to_keep = [(uid_mdn_0, bodyid) for bodyid in uids_down_VNC_graph]
specific_subgraph = VNC.subgraph(edges = edges_to_keep)
vnc_table = specific_subgraph.get_connections()
vnc_table.sort_values(by=':END_ID(Body-ID)', ascending=False, inplace=True)
print(
    f'Specific subgraph: ',
    vnc_table
    )

# Cmatrix derived from Connections object:
cm = VNC.get_cmatrix()
down_cm = cm.list_downstream_neurons(uid_mdn_0)
print(f'Downstream CMatrix (total {len(down_cm)})')

# traced_connections
traced_table = pd.read_csv(params.CONNECTIONS_FILE)
traced_table = traced_table[(
    traced_table['bodyId_pre'] == mdn_0
    )]
traced_table = traced_table[(
    traced_table['weight'] >= 5
    )]
traced_table.sort_values(by='bodyId_post', ascending=False, inplace=True)
print(f'Traced connections (total {len(traced_table)})')
print(traced_table)

# overlap
overlap = list(set(traced_table['bodyId_post']).intersection(set(down_VNC_graph)))
print('Overlap between both datasets:', len(overlap))





