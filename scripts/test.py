import pandas as pd
import os
import params
import numpy as np

synapses = pd.read_feather(
    os.path.join(
        params.NEUPRINT_RAW_DIR,
        "Neuprint_Synapses_manc_v1.ftr"
    )
)
'''
row = data[data[':ID(Syn-ID)'] == 99000000001]
for c_ in data.columns:
    print(f"{c_}: {row[c_].values[0]}")
    print('\n')


synapses = pd.read_feather(
    params.NEUPRINT_SYNAPSSET_FILE
    )

'''

roi_file = os.path.join(
            params.NEUPRINT_RAW_DIR,
            'all_ROIs.txt'
        )
rois = list(pd.read_csv(roi_file, sep='\t').values.flatten())
potential_column_names = [roi + ':boolean' for roi in rois]

# Function to find the column name with the True value
def find_true_value(row):
    true_columns = row[row == True].index
    return true_columns[0].replace(':boolean', '') if len(true_columns) > 0 else None

# Apply the function to each row

# for each row, find the column in potential_column_names that has a True value
# and add the corresponding roi to the 'neuropil' column
synapses['neuropil'] = synapses[
    potential_column_names
    ].apply(find_true_value, axis=1)
print(synapses.head())