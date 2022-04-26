import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# Function to write the log to file
def print_f(fptr, print_str, with_date=True):

    if with_date:
        fptr.write(str(datetime.now()) + ': ' + str(print_str) + '\n')
    else:
        fptr.write(str(print_str) + '\n')

    fptr.flush()
    os.fsync(fptr.fileno())


# Write selected orbits to file
def select_orbits(logs, files, split):

    with open(f'{logs}/{split}.txt', 'w') as orbits:
        for f in files:
            orbits.write(f + '\n')


# Load orbits listed in file
def load_data(orbits_file, orbits_file2=None):

    df_list = []
    with open(orbits_file, 'r') as orbits:
        files = orbits.read().splitlines()
    
    files2 = []
    if orbits_file2:
        with open(orbits_file2, 'r') as orbits:
            files2 = orbits.read().splitlines()
    
    files += files2
    for f in files:
        df_orbit = pd.read_csv(f, index_col=None, header=0).dropna()
        df_list.append(df_orbit)

    df = pd.concat(df_list, axis=0, ignore_index=True)
    df['SPLIT'] = orbits_file.split('/')[-1].split('.')[0]

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


# Plot all orbits with crossings
def plot_orbit(logs, df, pred=False, draw=[1, 3]):

    df['B_tot'] = (df['BX_MSO']**2 + df['BY_MSO']**2 + df['BZ_MSO']**2)**0.5
    colours = {0: 'red', 1: 'green', 2: 'yellow', 3: 'blue', 4: 'purple'}

    title = ' labels in training orbit '
    if df['SPLIT'][0] == 'test':
        title = ' labels in testing orbit '

    folder = df['SPLIT'][0] + '-'
    label_col = 'LABEL'
    if pred:
        label_col = 'LABEL_PRED'
        title = 'Preicted' + title
        folder += 'pred'
    else:
        title = 'True' + title
        folder += 'true'

    if not os.path.exists(f'{logs}/{folder}'):
        os.makedirs(f'{logs}/{folder}')

    for orbit in np.unique(df['ORBIT']):

        df_orbit = df.loc[df['ORBIT'] == orbit]
        fig = go.Figure()

        # Plotting components of the magnetic field B_x, B_y, B_z in MSO coordinates
        fig.add_trace(go.Scatter(
            x=df_orbit['DATE'], y=df_orbit['BX_MSO'], name='B_x'))
        fig.add_trace(go.Scatter(
            x=df_orbit['DATE'], y=df_orbit['BY_MSO'], name='B_y'))
        fig.add_trace(go.Scatter(
            x=df_orbit['DATE'], y=df_orbit['BZ_MSO'], name='B_z'))

        # Plotting total magnetic field magnitude B along the orbit
        fig.add_trace(go.Scatter(
            x=df_orbit['DATE'], y=-df_orbit['B_tot'], name='|B|', line_color='darkgray'))
        fig.add_trace(go.Scatter(x=df_orbit['DATE'], y=df_orbit['B_tot'], name='|B|',
                                 line_color='darkgray', showlegend=False))

        for i in draw:
            for _, row in df_orbit[df_orbit[label_col] == i].iterrows():
                fig.add_trace(go.Scatter(
                    x=[row['DATE'], row['DATE']],
                    y=[-450, 450],
                    mode='lines',
                    line_color=colours[i],
                    opacity=0.1,
                    showlegend=False
                ))

        fig.update_layout({'title': f'{title + str(orbit)}'})
        fig.write_image(
            f'{logs}/{folder}/fig_{orbit}.png')
        # fig.write_html(
        #     f'{logs}/{folder}/fig_{orbit}.png')


# logs = 'logs/123'
# if not os.path.exists(logs):
#     os.makedirs(logs)
# files = glob.glob('data/orbits/*.csv')
# files.sort(key=lambda x: int(''.join(i for i in x if i.isdigit())))

# select_orbits(logs, files[:3], 'train')
# select_orbits(logs, files[3:], 'test')

# df_train = load_data(f'{logs}/train.txt')
# df_test = load_data(f'{logs}/test.txt')

# df_train1 = select_features(df_train, 'data/features.txt')
# df_test1 = select_features(df_test, 'data/features.txt')

# plot_orbit(logs, df_train)
# plot_orbit(logs, df_test)
