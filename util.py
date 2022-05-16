import glob
import os
from datetime import datetime

import pandas as pd


# Function to write the log to file
def print_f(fptr, print_str, with_date=True):

    if with_date:
        fptr.write(str(datetime.now()) + ': ' + str(print_str) + '\n')
    else:
        fptr.write(str(print_str) + '\n')

    fptr.flush()
    os.fsync(fptr.fileno())


# Load orbits listed in file
def load_data(files, add_known_drifts=False):

    df_list = []

    # Load known orbits from a separate directory
    if add_known_drifts:
        files_known = glob.glob('data/drifts/*.csv')
        files_known.sort(key=lambda x: int(
            ''.join(i for i in x if i.isdigit())))
        for f in files_known:
            df_orbit = pd.read_csv(f, index_col=None, header=0).dropna()
            df_list.append(df_orbit)

    for f in files:
        df_orbit = pd.read_csv(f, index_col=None, header=0).dropna()
        df_list.append(df_orbit)

    df = pd.concat(df_list, axis=0, ignore_index=True)

    return df


# Select features listed in a file
def select_features(df, features_file):

    with open(features_file, 'r') as features:
        cols = features.read().splitlines()

    drop_col = ['Unnamed: 0', 'X_MSO', 'Y_MSO', 'Z_MSO', 'BX_MSO', 'BY_MSO', 'BZ_MSO', 'DBX_MSO', 'DBY_MSO', 'DBZ_MSO', 'RHO_DIPOLE', 'PHI_DIPOLE', 'THETA_DIPOLE',
                'BABS_DIPOLE', 'BX_DIPOLE', 'BY_DIPOLE', 'BZ_DIPOLE', 'RHO', 'RXY', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'VABS', 'D', 'COSALPHA', 'EXTREMA']

    for col in cols:
        if col in drop_col:
            drop_col.remove(col)

    if 'INDEX' in cols:
        drop_col.remove('Unnamed: 0')

    return df.drop(drop_col, axis=1)


# Load orbits and drifts from file
def load_drifts(drifts_file):

    drift_orbits = {}
    with open(drifts_file, 'r') as drifts:
        for line in drifts.read().splitlines():
            line = list(map(int, line.split(' ')))
            drift_orbits[line[0]] = line[1]

    return drift_orbits
