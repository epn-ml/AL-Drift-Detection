# %% imports

import glob
import os
import sys
from datetime import datetime
from time import perf_counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from skmultiflow.trees import HoeffdingTreeClassifier

global fptr
global folder


# %% functions

# Function to write the log to disk also
def print_(print_str):
    global fptr
    if fptr is None:
        os.makedirs(f'../logs/{folder}')
        name = f'../logs/{folder}/log.txt'
        fptr = open(name, "w")

    fptr.write(str(datetime.now()) + ": " + str(print_str) + "\n")
    print(str(datetime.now()) + ": " + str(print_str))
    fptr.flush()
    os.fsync(fptr.fileno())


def fit_and_predict(clf, features, labels, classes):
    predicted = np.empty(shape=len(labels))
    predicted[0] = clf.predict([features[0]])
    clf.reset()
    clf.partial_fit([features[0]], [labels[0]], classes=classes)
    for idx in range(1, len(labels)):
        predicted[idx] = clf.predict([features[idx]])
        clf.partial_fit([features[idx]], [labels[idx]], classes=classes)

    return predicted, clf


def predict_and_partial_fit(clf, features, labels, classes):
    predicted = np.empty(shape=len(labels))
    for idx in range(0, len(labels)):
        predicted[idx] = clf.predict([features[idx]])
        clf.partial_fit([features[idx]], [labels[idx]], classes=classes)

    return predicted, clf


def load_data(path):

    files = glob.glob(path)
    li = []
    breaks = []

    for filename in files:
        df = pd.read_csv(filename, index_col=None, header=0)
        breaks.append((df.iloc[0]['DATE'], df.iloc[-1]['DATE']))
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)
    print_(f'loaded data: {files}')

    return df.dropna().sort_values(by='DATE'), sorted(breaks)


def select_features(df, features):

    drop_col = ['Unnamed: 0', 'X_MSO', 'Y_MSO', 'Z_MSO', 'BX_MSO', 'BY_MSO', 'BZ_MSO', 'DBX_MSO', 'DBY_MSO', 'DBZ_MSO', 'RHO_DIPOLE', 'PHI_DIPOLE', 'THETA_DIPOLE',
                'BABS_DIPOLE', 'BX_DIPOLE', 'BY_DIPOLE', 'BZ_DIPOLE', 'RHO', 'RXY', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'VABS', 'D', 'COSALPHA', 'EXTREMA', 'ORBIT']

    for feature in features:
        if feature in drop_col:
            drop_col.remove(feature)

    if 'INDEX' in features:
        drop_col.remove('Unnamed: 0')

    return df.drop(drop_col, axis=1)


def plot_field(df):
    fig = go.Figure()

    # Plotting components of the magnetic field B_x, B_y, B_z in MSO coordinates
    fig.add_trace(go.Scatter(x=df['DATE'], y=df['BX_MSO'], name='B_x'))
    fig.add_trace(go.Scatter(x=df['DATE'], y=df['BY_MSO'], name='B_y'))
    fig.add_trace(go.Scatter(x=df['DATE'], y=df['BZ_MSO'], name='B_z'))

    # Plotting total magnetic field magnitude B along the orbit
    fig.add_trace(go.Scatter(
        x=df['DATE'], y=-df['B_tot'], name='|B|', line_color='darkgray'))
    fig.add_trace(go.Scatter(x=df['DATE'], y=df['B_tot'], name='|B|',
                  line_color='darkgray', showlegend=False))

    return fig


def plot_orbit(df, breaks, title, draw=[1, 3], labels=None):

    df['B_tot'] = (df['BX_MSO']**2 + df['BY_MSO']**2 + df['BZ_MSO']**2)**0.5
    colors = {0: 'red', 1: 'green', 2: 'yellow', 3: 'blue', 4: 'purple'}

    label_col = 'LABEL'
    if labels is not None:
        df['LABEL_PRED'] = labels
        label_col = 'LABEL_PRED'

    for date_range in breaks:

        df_orbit = df[(df['DATE'] >= date_range[0]) &
                      (df['DATE'] <= date_range[1])]
        fig = plot_field(df_orbit)

        for i in draw:
            for _, row in df_orbit[df_orbit[label_col] == i].iterrows():
                fig.add_trace(go.Scatter(
                    x=[row['DATE'], row['DATE']],
                    y=[-450, 450],
                    mode='lines',
                    line_color=colors[i],
                    opacity=0.05,
                    showlegend=False
                ))

        fig.update_layout({'title': title})
        fig.write_html(
            f'../logs/{folder}/fig_{df_orbit.iloc[0]["DATE"][:16].replace(" ", "_").replace(":", "-")}_{title}.html')


# %% setup

fptr = None
folder = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

# Set the number of training instances
training_window_size = 1000
if len(sys.argv) == 2:
    training_window_size = int(sys.argv[1])
print_(f'training_window_size: {training_window_size}')


# %% load data

df_train, breaks_train = load_data('../data/orbits/train/*.csv')
df_test, breaks_test = load_data('../data/orbits/test/*.csv')


# %% select data

feats = ['X_MSO', 'Y_MSO', 'Z_MSO', 'BX_MSO', 'BY_MSO', 'BZ_MSO', 'DBX_MSO', 'DBY_MSO', 'DBZ_MSO', 'RHO_DIPOLE', 'PHI_DIPOLE', 'THETA_DIPOLE',
         'BABS_DIPOLE', 'BX_DIPOLE', 'BY_DIPOLE', 'BZ_DIPOLE', 'RHO', 'RXY', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'VABS', 'D', 'COSALPHA', 'EXTREMA']

with open('../data/features.txt', 'r') as f:
    feats = [line.strip() for line in f]

df_train = select_features(df_train, feats)
df_test = select_features(df_test, feats)
print_(f'selected features: {feats}')

# offset_train = 16080
# size_train = 26280 - offset_train
# offset_test = 18891
# size_test = 27291 - offset_test
# print(f'offset_train: {offset_train}, size_train: {size_train}')
# print(f'offset_test: {offset_test}, size_test: {size_test}')

# df_train = df_train.iloc[offset_train:offset_train+size_train]
# df_test = df_test.iloc[offset_test:offset_test+size_test]

features_train = df_train.iloc[:, 1:-1].values
labels_train = df_train.iloc[:, -1].values.tolist()
mean = np.mean(features_train, axis=1).reshape(features_train.shape[0], 1)
std = np.std(features_train, axis=1).reshape(features_train.shape[0], 1)
features_train = (features_train - mean)/(std + 0.000001)
u, c = np.unique(labels_train, return_counts=True)
print_(dict(zip(u, c)))
print_(f'features_train: {len(features_train)}')

features_test = df_test.iloc[:, 1:-1].values
labels_test = df_test.iloc[:, -1].values.tolist()
mean = np.mean(features_test, axis=1).reshape(features_test.shape[0], 1)
std = np.std(features_test, axis=1).reshape(features_test.shape[0], 1)
features_test = (features_test - mean)/(std + 0.000001)
u, c = np.unique(labels_test, return_counts=True)
print_(dict(zip(u, c)))
print_(f'features_test: {len(features_test)}')


# %% training

"""
# Min max scaling
min_features = np.min(features, axis=1)
features = features - np.reshape(min_features, newshape=(min_features.shape[0], 1))
max_features = np.max(features, axis=1)
max_features = np.reshape(max_features, newshape=(max_features.shape[0], 1)) + 0.000001
features = features / max_features
"""

print_('training...')
t1 = perf_counter()
train_pred, clf = fit_and_predict(
    clf=HoeffdingTreeClassifier(), features=features_train, labels=labels_train, classes=np.unique(labels_train))
t2 = perf_counter()

train_pred = train_pred.tolist()
train_true = labels_train
test_true = labels_test


# %% testing

print_('testing...')
test_pred = np.empty(shape=len(features_test))
for idx in range(0, len(features_test)):
    test_pred[idx] = clf.predict([features_test[idx]])

# %% pad missing labels

if len(train_true) < len(df_train.index):
    print_(
        f'padding training set true values with [{train_true[-1]}] * {len(df_train.index) - len(train_true)}')
    train_true += [train_true[-1]] * (len(df_train.index) - len(train_true))
if len(train_pred) < len(df_train.index):
    print_(
        f'padding training set predictions with [{train_pred[-1]}] * {len(df_train.index) - len(train_pred)}')
    train_pred += [train_pred[-1]] * (len(df_train.index) - len(train_pred))

print_(f'training set size: {len(train_true)}')
print_(f'testing set size: {len(test_true)}')

# %% evaluation

auc_value = accuracy_score(y_true=train_true, y_pred=train_pred)
print_('Accuracy value is %f for training dataset %s' % auc_value)
print_(precision_recall_fscore_support(train_true, train_pred,
       average=None, labels=np.unique(train_true)))
print_(confusion_matrix(train_true, train_pred))

auc_value = accuracy_score(y_true=test_true, y_pred=test_pred)
print_('Accuracy value is %f for testing dataset %s' % auc_value)
print_(precision_recall_fscore_support(
    test_true, test_pred, average=None, labels=np.unique(test_true)))
print_(confusion_matrix(test_true, test_pred))

exec_time = t2 - t1
print_('Execution time is %d seconds' % exec_time)


# %% plots

print_('plotting...')
plot_orbit(df_train, breaks_train, 'train-true')
plot_orbit(df_train, breaks_train, 'train-pred', labels=train_pred)
plot_orbit(df_test, breaks_test, 'test-true')
plot_orbit(df_test, breaks_test, 'test-pred', labels=test_pred)
print_('plotting finished')


# %% close log file

if fptr is not None:
    fptr.close()
    fptr = None

# %%
