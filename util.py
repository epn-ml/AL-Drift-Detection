import glob
import os
from datetime import datetime

import pandas as pd


def print_f(fptr, print_str, with_date=True):
    """
    Writes the log to file.

    Args:
        fptr: File-like object.
        print_str: Value to be printed.
        with_date: If ``True``, append current datetime before the value.
    """
    if with_date:
        fptr.write(str(datetime.now()) + ': ' + str(print_str) + '\n')
    else:
        fptr.write(str(print_str) + '\n')

    fptr.flush()
    os.fsync(fptr.fileno())


def load_data(files, add_known_drifts=False):
    """
    Loads orbits listed in file.

    Args:
        files: List of file paths to orbit data.
        add_known_drifts: If ``True``, load orbits with known drifts first.

    Returns:
        df: DataFrame with all loaded orbit data.
    """
    df_list = []

    # Load known orbits from a separate directory
    if add_known_drifts:
        files_known = glob.glob('data/drifts/*.csv')
        files_known.sort(key=lambda x: int(
            ''.join(i for i in x if i.isdigit())))
        for f in files_known:
            df_orbit = pd.read_csv(f, index_col=None, header=0).dropna()
            if not 'ORBIT' in df_orbit:
                orbit = int(f.split('_')[1].split['.'][0])
                df_orbit['ORBIT'] = orbit
            df_list.append(df_orbit)

    for f in files:
        df_orbit = pd.read_csv(f, index_col=None, header=0).dropna()
        if not 'ORBIT' in df_orbit:
            orbit = int(f.split('_')[1].split('.')[0])
            df_orbit['ORBIT'] = orbit
        df_list.append(df_orbit)

    df = pd.concat(df_list, axis=0, ignore_index=True)

    return df


def select_features(df, features_file):
    """
    Selects features listed in a file.

    Args:
        df: DataFrame with loaded orbit data.
        features_file: File with listed features for selection.

    Returns:
        df: Orbit with only selected features.
    """
    with open(features_file, 'r') as features:
        cols = features.read().splitlines()

    drop_col = ['Unnamed: 0',
                'X_MSO', 'Y_MSO', 'Z_MSO',
                'BX_MSO', 'BY_MSO', 'BZ_MSO',
                'DBX_MSO', 'DBY_MSO', 'DBZ_MSO',
                'RHO_DIPOLE', 'PHI_DIPOLE', 'THETA_DIPOLE',
                'BABS_DIPOLE', 'BX_DIPOLE', 'BY_DIPOLE', 'BZ_DIPOLE',
                'RHO', 'RXY',
                'X', 'Y', 'Z',
                'VX', 'VY', 'VZ', 'VABS',
                'D', 'COSALPHA', 'EXTREMA',
                'X_AB', 'Y_AB', 'Z_AB',
                'BX_AB', 'BY_AB', 'BZ_AB',
                'DBX_AB', 'DBY_AB', 'DBZ_AB',
                'RHO_AB', 'RXY_AB']

    for col in cols:
        if col in drop_col:
            drop_col.remove(col)

    if 'INDEX' in cols:
        drop_col.remove('Unnamed: 0')

    df = df.drop(drop_col, axis=1)
    if 'TIME_TAG' in df:
        df.drop('TIME_TAG', axis=1)
    if 'NAVG' in df:
        df.drop('NAVG', axis=1)

    return df


def load_drifts(drifts_file):
    """
    Loads orbits and drifts from file.

    Args:
        drifts_file: File with orbit numbers and drift labels.

    Returns:
        drift_orbits: Orbit numbers and corresponding drift labels.
    """
    drift_orbits = {}
    with open(drifts_file, 'r') as drifts:
        for line in drifts.read().splitlines():
            line = list(map(int, line.split(' ')))
            drift_orbits[line[0]] = line[1]

    return drift_orbits
