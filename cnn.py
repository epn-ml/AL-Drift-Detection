# %% imports

import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
import wandb
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers, metrics
from wandb.keras import WandbCallback

from util import load_data, load_drifts, print_f, select_features

global fptr


# %% Functions

# Wrapper for print function
def print_(print_str, with_date=True):

    global fptr
    print_f(fptr, print_str, with_date)
    if with_date:
        print(f'{str(datetime.now())}: {print_str}')
    else:
        print(print_str)


# Create CNN model
def cnn(shape):

    model = keras.Sequential()
    model.add(layers.Conv1D(64, 2, strides=1, activation='relu',
              padding='same', input_shape=shape))
    model.add(layers.LSTM(64, return_sequences=True))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(5, activation='softmax'))

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy', metrics.sparse_categorical_accuracy])

    return model


# Train classifier based on drift
def train_clf(df, max_orbits=10):

    # Standardization
    df_features = df.iloc[:, 1:-5]
    # print_(f'features:\n{df_features.head()}')

    df.iloc[:, 1:-5] = (df_features - df_features.mean()) / df_features.std()
    # print_(f'standardized:\n{df.iloc[:, 1:-5].head()}')
    print_(f'total size = {len(df.index)}')

    clfs = []
    len_features = len(df.iloc[:, 1:-5].columns)
    drifts = pd.unique(df['DRIFT']).tolist()
    print_(f'drifts: {drifts}')
    print_(f'========================================')

    for drift in drifts:

        wandb.log({"drift": drift})
        clf = cnn((len_features, 1))
        print_(f'cnn for drift {drift} ({len_features} features):')
        clf.summary(print_fn=print_)

        df_drift_train = df.loc[(df['DRIFT'] == drift)
                                & (df['SPLIT'] == 'train')]
        orbit_numbers = pd.unique(df_drift_train['ORBIT']).tolist()
        print_(f'train orbits with drift {drift}: {orbit_numbers}')

        x_train = []
        y_train = []

        for orbit in orbit_numbers:

            df_orbit = df_drift_train.loc[df['ORBIT'] == orbit]
            features = df_orbit.iloc[:, 1:-5].values
            labels = df_orbit['LABEL'].tolist()

            x = np.array(features, copy=True)
            x = x.reshape(-1, x.shape[1], 1)
            y = np.asarray(labels)

            x_train += x.tolist()
            y_train += y.tolist()

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        classes = np.unique(y_train)

        weights = compute_class_weight(
            'balanced', classes=classes, y=y_train)
        print_(f'weights: {weights}')

        clf.fit(x=x_train, y=y_train,
                batch_size=16,
                epochs=13,
                callbacks=[WandbCallback()],
                class_weight={k: v for k,
                              v in enumerate(weights)},
                verbose=2)

        clfs.append(clf)
        acc = clf.evaluate(x_train, y_train, verbose=2)
        print_(f'metric names: {clf.metrics_names}')
        print_(f'evaluation (drift {drift}): {acc}')

        # Training evaluation
        labels_pred = clf.predict(x_train)
        labels_pred = labels_pred.argmax(axis=-1)
        df.loc[(df['DRIFT'] == drift) & (df['SPLIT'] ==
                                         'train'), 'LABEL_PRED'] = labels_pred
        prf = precision_recall_fscore_support(
            y_true=y_train, y_pred=labels_pred, average=None, labels=classes)
        print_(f'training precision: {prf[0]}')
        print_(f'training recall: {prf[1]}')
        print_(f'training f-score: {prf[2]}')

        # Testing
        df_drift_test = df.loc[(df['DRIFT'] == drift) &
                               (df['SPLIT'] == 'test')]
        orbit_numbers_test = pd.unique(df_drift_test['ORBIT']).tolist()
        print_(f'{len(orbit_numbers_test)} test orbits with drift {drift}')
        print_(f'selected orbits for testing: {orbit_numbers_test}')

        features_test = df_drift_test.iloc[:, 1:-5].values
        x_test = np.array(features_test, copy=True)
        x_test = x_test.reshape(-1, x_test.shape[1], 1)

        # Testing evaluation
        labels_pred_test = clf.predict(x_test)
        labels_pred_test = labels_pred_test.argmax(axis=-1)
        df.loc[(df['DRIFT'] == drift) & (df['SPLIT'] == 'test'),
               'LABEL_PRED'] = labels_pred_test
        y_test = df_drift_test['LABEL'].tolist()
        prf_test = precision_recall_fscore_support(
            y_true=y_test, y_pred=labels_pred_test, average=None, labels=classes)
        print_(f'testing precision: {prf_test[0]}')
        print_(f'testing recall: {prf_test[1]}')
        print_(f'testing f-score: {prf_test[2]}')

        print_(f'========================================')

    return df


# Test classifier
def test_clfs(df, clf):

    # Standardization
    df_features = df.iloc[:, 1:-5]
    print_(f'features:\n{df_features.head()}')

    df.iloc[:, 1:-5] = (df_features - df_features.mean()) / df_features.std()
    print_(f'standardized:\n{df.iloc[:, 1:-5].head()}')
    print_(f'total size = {len(df.index)}')

    features = df.iloc[:, 1:-5].values
    x = np.array(features, copy=True)
    x = x.reshape(-1, x.shape[1], 1)

    labels_pred = clf.predict(x)  # window vs step
    labels_pred = labels_pred.argmax(axis=-1)
    df['LABEL_PRED'] = labels_pred

    return df['LABEL_PRED']


# Plot all orbits with crossings
def plot_orbits(logs, dataset, df, test=False, pred=False, draw=[1, 3]):

    colours = {0: 'red', 1: 'green', 2: 'yellow', 3: 'blue', 4: 'purple'}
    title = 'labels in training orbit '
    folder = 'train-'
    if test:
        title = 'labels in testing orbit '
        folder = 'test-'
        df = df.loc[df['SPLIT'] == 'test']
    else:
        df = df.loc[df['SPLIT'] == 'train']

    label_col = 'LABEL'
    if pred:
        label_col = 'LABEL_PRED'
        title = 'Preicted ' + title
        folder += 'pred'
    else:
        title = 'True ' + title
        folder += 'true'

    if not os.path.exists(f'{logs}/plots_set{dataset}/{folder}'):
        os.makedirs(f'{logs}/plots_set{dataset}/{folder}')

    for orbit in pd.unique(df['ORBIT']).tolist():

        df_orbit = df.loc[df['ORBIT'] == orbit]
        fig = go.Figure()

        # Plotting components of the magnetic field B_x, B_y, B_z in MSO coordinates
        # fig.add_trace(go.Scatter(
        #     x=df_orbit['DATE'], y=df_orbit['BX_MSO'], name='B_x'))
        # fig.add_trace(go.Scatter(
        #     x=df_orbit['DATE'], y=df_orbit['BY_MSO'], name='B_y'))
        fig.add_trace(go.Scatter(
            x=df_orbit['DATE'], y=df_orbit['BZ_MSO'], name='B_z'))
        fig.add_trace(go.Scatter(
            x=df_orbit['DATE'], y=df_orbit['COSALPHA'], name='cos_a'))

        # Plotting total magnetic field magnitude B along the orbit
        # fig.add_trace(go.Scatter(
        #     x=df_orbit['DATE'], y=-df_orbit['B_tot'], name='|B|', line_color='darkgray'))
        fig.add_trace(go.Scatter(x=df_orbit['DATE'], y=df_orbit['B_tot'], name='|B|',
                                 line_color='darkgray', showlegend=False))

        for i in draw:
            for _, row in df_orbit.loc[df_orbit[label_col] == i].iterrows():
                fig.add_trace(go.Scatter(
                    x=[row['DATE'], row['DATE']],
                    y=[-450, 450],
                    mode='lines',
                    line_color=colours[i],
                    opacity=0.1,
                    showlegend=False
                ))

        fig.update_layout(
            {'title': f'{title}{orbit} (drift {df_orbit.iloc[0]["DRIFT"]})'})
        fig.write_image(
            f'{logs}/plots_set{dataset}/{folder}/fig{orbit}_drift{df_orbit.iloc[0]["DRIFT"]}.png')
        # fig.write_html(
        #     f'{logs}/{folder}/fig_{orbit}.png')


# %% Setup

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print_(e)

wandb.init(project="cnn", entity="irodionr", config={
    "filters": 64,
    "kernel_size": 2,
    "units_lstm": 64,
    "units_dense": 16,
    "batch_size": 128,
    "epochs": 13,
    "learning_rate": 0.001
})

logs = sys.argv[1]
dataset = int(sys.argv[2])
plots = sys.argv[3]
# logs = 'logs_cnn'
# dataset = 1
# plots = '0123'
if not os.path.exists(logs):
    os.makedirs(logs)

fptr = open(f'{logs}/log_cnn_set{dataset}.txt', 'w')
print_(f'dataset: {dataset}')


# %% Load data

drift_orbits = load_drifts(f'data/drifts_set{dataset}.txt')
files = []
cur_orbit = 0
for orb in drift_orbits:
    if cur_orbit < 100:
        files.append(f'data/drifts/df_{orb}.csv')
        print_(f'data/drifts/df_{orb}.csv', with_date=False)
    else:
        files.append(f'data/orbits/df_{orb}.csv')
        print_(f'data/orbits/df_{orb}.csv', with_date=False)
    cur_orbit += 1

df = load_data(files)
df = select_features(df, 'data/features_cnn.txt')

df['LABEL_PRED'] = 0
df['DRIFT'] = 1
df['SPLIT'] = 'train'

# Randomly select orbits for testing
len_train = 0
len_test = 0
max_orbits = 10
for drift in pd.unique(list(drift_orbits.values())).tolist():

    all_orbits = [k for k, v in drift_orbits.items() if v == drift]
    train_count = len(all_orbits) * 4 // 5  # 80% of orbits
    if train_count == 0:
        train_count = len(all_orbits) - 1
        if train_count == 0:
            train_count = 1
    if train_count > max_orbits:
        train_count = max_orbits

    if train_count != len(all_orbits):
        train_orbits = [
            all_orbits[i] for i in sorted(random.sample(range(len(all_orbits)), train_count))
        ]
    else:
        train_orbits = all_orbits

    test_orbits = [orb for orb in all_orbits if orb not in train_orbits]
    len_train += len(train_orbits)
    len_test += len(test_orbits)

    print_(f'train orbits for drift {drift}: {train_orbits}')
    for orb in train_orbits:
        df.loc[df['ORBIT'] == orb, 'DRIFT'] = drift
    print_(f'test orbits for drift {drift}: {test_orbits}')
    for orb in test_orbits:
        df.loc[df['ORBIT'] == orb, 'DRIFT'] = drift
        df.loc[df['ORBIT'] == orb, 'SPLIT'] = 'test'

# print_(f'selected data:\n{df.head()}')
print_(f'total train orbits: {len_train}')
print_(f'total test orbits: {len_test}')


# %% Training classifiers

t1 = time.perf_counter()
df = train_clf(df.copy())
t2 = time.perf_counter()
print_(f'training time is {t2 - t1:.2f} seconds')


# %% Testing classifiers

# t1 = time.perf_counter()
# df_pred = test_clfs(df.copy(), clf)
# t2 = time.perf_counter()
# print_(f'testing time is {t2 - t1:.2f} seconds')

# df['LABEL_PRED'] = df_pred


# %% Evaluation

drifts = pd.unique(df['DRIFT']).tolist()
print_(f'drifts: {drifts}')

for drift in drifts:

    df_drift = df.loc[df['DRIFT'] == drift]
    orbit_numbers = pd.unique(df_drift['ORBIT']).tolist()
    print_(f'{len(orbit_numbers)} orbits with drift {drift}')
    print_(f'{orbit_numbers}')

    for orbit in orbit_numbers:

        df_orbit = df_drift.loc[df['ORBIT'] == orbit]
        labels = df_orbit['LABEL'].tolist()
        labels_pred = df_orbit['LABEL_PRED'].tolist()
        classes = np.unique(labels)

        prf = precision_recall_fscore_support(
            y_true=labels, y_pred=labels_pred, average=None, labels=classes)
        print_(
            f'{df_orbit.iloc[0]["SPLIT"]} orbit {orbit} f-score: {prf[2]}, recall: {prf[1]}, precision: {prf[0]}')

labels_train_true = df.loc[df['SPLIT'] == 'train', 'LABEL'].tolist()
labels_train_pred = df.loc[df['SPLIT'] == 'train', 'LABEL_PRED'].tolist()
labels_test_true = df.loc[df['SPLIT'] == 'test', 'LABEL'].tolist()
labels_test_pred = df.loc[df['SPLIT'] == 'test', 'LABEL_PRED'].tolist()

auc_value = accuracy_score(y_true=labels_train_true, y_pred=labels_train_pred)
print_(f'accuracy value is {auc_value} for training dataset')
prf = precision_recall_fscore_support(
    labels_train_true, labels_train_pred, average=None, labels=np.unique(labels_train_true))
print_(f'precision: {prf[0]}')
print_(f'recall: {prf[1]}')
print_(f'f-score: {prf[2]}')
print_(f'support: {prf[3]}')
print_(
    f'confusion matrix:\n{confusion_matrix(labels_train_true, labels_train_pred)}')

auc_value = accuracy_score(y_true=labels_test_true, y_pred=labels_test_pred)
print_(f'accuracy value is {auc_value} for testing dataset')
prf = precision_recall_fscore_support(
    labels_test_true, labels_test_pred, average=None, labels=np.unique(labels_test_true))
print_(f'precision: {prf[0]}')
print_(f'recall: {prf[1]}')
print_(f'f-score: {prf[2]}')
print_(f'support: {prf[3]}')
print_(
    f'confusion matrix:\n{confusion_matrix(labels_test_true, labels_test_pred)}')


# %% Plots

if plots != '':
    df['B_tot'] = (df['BX_MSO']**2 + df['BY_MSO']**2 + df['BZ_MSO']**2)**0.5
    print_(f'plotting {plots}...')
    if '0' in plots:
        plot_orbits(logs, dataset, df, test=False, pred=False)
        print_(f'plotted train-true')
    if '1' in plots:
        plot_orbits(logs, dataset, df, test=False, pred=True)
        print_(f'plotted train-pred')
    if '2' in plots:
        plot_orbits(logs, dataset, df, test=True, pred=False)
        print_(f'plotted test-true')
    if '3' in plots:
        plot_orbits(logs, dataset, df, test=True, pred=True)
        print_(f'plotted test-pred')


# %% Close log file

if fptr is not None:
    fptr.close()
    fptr = None
