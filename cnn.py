# %% imports

import os
import random
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
import wandb
from keras.callbacks import Callback
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers
from wandb.keras import WandbCallback

from util import load_data, load_drifts, print_f, select_features

global fptr


class Metrics(Callback):

    def on_train_begin(self, logs={}):
        self.val_auc = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1s = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(
            self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_auc = accuracy_score(y_true=val_targ, y_pred=val_predict)
        prf = precision_recall_fscore_support(
            val_targ, val_predict, average=None, labels=np.unique(val_targ))
        _val_precision = prf[0]
        _val_recall = prf[1]
        _val_f1 = prf[2]
        self.val_auc.append(_val_auc)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)

        print_(f'epoch: {epoch}')
        print_(f'precision: {prf[0]}')
        print_(f'recall: {prf[1]}')
        print_(f'f-score: {prf[2]}')
        print_(f'support: {prf[3]}')

        return


# %% Functions

# Wrapper for print function
def print_(print_str, with_date=True):

    global fptr
    print_f(fptr, print_str, with_date)


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
                  metrics=['accuracy'])

    return model


# Train classifier based on drift
def train_clf(df, max_orbits=5):

    # Standardization
    df_features = df.iloc[:, 1:-5]
    print_(f'features:\n{df_features.head()}')

    df.iloc[:, 1:-5] = (df_features - df_features.mean()) / df_features.std()
    print_(f'standardized:\n{df.iloc[:, 1:-5].head()}')
    print_(f'total size = {len(df.index)}')

    len_features = len(df.iloc[:, 1:-5].columns)
    clf = cnn((len_features, 1))
    print_(f'cnn ({len_features} features):')
    clf.summary(print_fn=print_)

    drifts = pd.unique(df['DRIFT']).tolist()
    print_(f'drifts: {drifts}')

    x_train = []
    y_train = []

    for drift in drifts:

        df_drift = df.loc[df['DRIFT'] == drift]
        orbit_numbers = pd.unique(df_drift['ORBIT']).tolist()
        print_(f'{len(orbit_numbers)} train orbits with drift {drift}')
        if len(orbit_numbers) > max_orbits:
            random.shuffle(orbit_numbers)
            orbit_numbers = orbit_numbers[:max_orbits]
        print_(f'selected orbits for training: {orbit_numbers}')

        for orbit in orbit_numbers:

            df_orbit = df_drift.loc[df['ORBIT'] == orbit]
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

    wandb.config = {
        'filters': 64,
        'kernel_size': 2,
        'units_lstm': 64,
        'units_dense': 16,
        'batch_size': 128,
        'epochs': 20,
        'learning_rate': 0.001
    }

    metrics = Metrics()

    clf.fit(x=x_train, y=y_train,
            validation_split=0.2,
            batch_size=16,
            epochs=20,
            callbacks=[WandbCallback(), metrics],
            class_weight={k: v for k,
                          v in enumerate(weights)},
            verbose=2)

    acc = clf.evaluate(x_train, y_train, verbose=2)
    print_(f'metric names: {clf.metrics_names}')
    print_(f'evaluation: {acc}')

    # Intermediate evaluation
    labels_pred = clf.predict(x_train)
    labels_pred = labels_pred.argmax(axis=-1)
    prf = precision_recall_fscore_support(
        y_true=y_train, y_pred=labels_pred, average=None, labels=classes)
    print_(f'training precision: {prf[0]}')
    print_(f'training recall: {prf[1]}')
    print_(f'training f-score: {prf[2]}')

    return clf


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
def plot_orbits(logs, df, test=False, pred=False, draw=[1, 3]):

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

    if not os.path.exists(f'{logs}/{folder}'):
        os.makedirs(f'{logs}/{folder}')

    for orbit in pd.unique(df['ORBIT']).tolist():

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
            f'{logs}/{folder}/fig{orbit}_drift{df_orbit.iloc[0]["DRIFT"]}.png')
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

wandb.init(project="cnn", entity="irodionr")

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
for drift in np.unique(list(drift_orbits.values())):

    all_orbits = [k for k, v in drift_orbits.items() if v == drift]
    test_count = len(all_orbits) // 5
    if test_count == 0:
        test_count = 1

    test_orbits = random.sample(all_orbits, test_count)
    train_orbits = [orb for orb in all_orbits if orb not in test_orbits]
    len_train += len(train_orbits)
    len_test += len(test_orbits)

    print_(f'train orbits for drift {drift}: {train_orbits}')
    for orb in train_orbits:
        df.loc[df['ORBIT'] == orb, 'DRIFT'] = drift
    print_(f'test orbits for drift {drift}: {test_orbits}')
    for orb in test_orbits:
        df.loc[df['ORBIT'] == orb, 'DRIFT'] = drift
        df.loc[df['ORBIT'] == orb, 'SPLIT'] = 'test'

print_(f'selected data:\n{df.head()}')
print_(f'total train orbits: {len_train}')
print_(f'total test orbits: {len_test}')


# %% Training classifiers

t1 = time.perf_counter()
clf = train_clf(df.loc[df['SPLIT'] == 'train'].copy())
t2 = time.perf_counter()
print_(f'training time is {t2 - t1:.2f} seconds')


# %% Testing classifiers

t1 = time.perf_counter()
df_pred = test_clfs(df.copy(), clf)
t2 = time.perf_counter()
print_(f'testing time is {t2 - t1:.2f} seconds')

df['LABEL_PRED'] = df_pred


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

        f1 = precision_recall_fscore_support(
            y_true=labels, y_pred=labels_pred, average=None, labels=classes)[2]
        print_(f'{df_orbit.iloc[0]["SPLIT"]} orbit {orbit} f-score: {f1}')

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
        plot_orbits(logs, df, test=False, pred=False)
        print_(f'plotted train-true')
    if '1' in plots:
        plot_orbits(logs, df, test=False, pred=True)
        print_(f'plotted train-pred')
    if '2' in plots:
        plot_orbits(logs, df, test=True, pred=False)
        print_(f'plotted test-true')
    if '3' in plots:
        plot_orbits(logs, df, test=True, pred=True)
        print_(f'plotted test-pred')


# %% Close log file

if fptr is not None:
    fptr.close()
    fptr = None
