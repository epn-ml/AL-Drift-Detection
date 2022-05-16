# %% imports

import os
import random
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

from util import load_data, load_drifts, print_f, select_features

global fptr


# %% Functions

# Wrapper for print function
def print_(print_str, with_date=True):

    global fptr
    print_f(fptr, print_str, with_date)


# Create CNN model
def cnn(shape):
    model = keras.Sequential()
    model.add(layers.Conv1D(64, 3, activation='relu', input_shape=shape))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.MaxPooling1D())
    model.add(layers.Flatten())
    model.add(layers.Dense(5, activation='softmax'))

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                  optimizer='adam', metrics=['accuracy'])

    return model


# Train classifier based on drift
def train_clf(df, max_count=10):

    # Standardization
    df_features = df.iloc[:, 1:-4]
    print_(f'features:\n{df_features.head()}')
    print_(f'mean:\n{df_features.mean()}')

    df.iloc[:, 1:-4] = (df_features - df_features.mean()) / df_features.std()
    print_(f'standardized:\n{df.iloc[:, 1:-4].head()}')
    print_(f'mean:\n{df.iloc[:, 1:-4].mean()}')
    print_(f'total size = {len(df.index)}')

    clf = cnn((len(df_features.columns), 1))
    print_(f'cnn:\n{clf.summary()}')

    for drift in pd.unique(df['DRIFT']).tolist():

        df_drift = df.loc[df['DRIFT'] == drift]
        orbit_numbers = pd.unique(df_drift['ORBIT']).tolist()
        print_(f'{len(orbit_numbers)} orbits with drift {drift}')
        if len(orbit_numbers) > max_count:
            random.shuffle(orbit_numbers)
            orbit_numbers = orbit_numbers[:max_count]
        print_(f'selected orbits for training: {orbit_numbers}')

        for orbit in orbit_numbers:

            df_orbit = df_drift.loc(df['ORBIT'] == orbit)
            features = df_orbit.iloc[:, 1:-4].values
            labels = df_orbit['LABEL'].tolist()
            classes = np.unique(labels)
            weights = compute_class_weight(
                'balanced', classes=classes, y=labels)
            print_(f'weights = {weights}')

            x = np.array(features, copy=True)
            x = x.reshape(-1, x.shape[1], 1)
            y = np.asarray(labels)

            print_(f'training classifier on orbit {orbit}')
            clf.fit(x=x, y=y,
                    batch_size=64,
                    epochs=20,
                    class_weight={k: v for k,
                                  v in enumerate(weights)},
                    verbose=0)

    print_(f'cnn:\n{clf.summary()}')

    return clf


# Test classifier
def test_clfs(df, clf):

    df['LABEL_PRED'] = 0

    # Standardization
    df_features = df.iloc[:, 1:-5]
    print_(f'features:\n{df_features.head()}')
    print_(f'mean:\n{df_features.mean()}')

    df.iloc[:, 1:-5] = (df_features - df_features.mean()) / df_features.std()
    print_(f'standardized:\n{df.iloc[:, 1:-5].head()}')
    print_(f'mean:\n{df.iloc[:, 1:-5].mean()}')
    print_(f'total size = {len(df.index)}')

    orbit_numbers = pd.unique(df['ORBIT']).tolist()
    for orbit in orbit_numbers:

        df_orbit = df.loc(df['ORBIT'] == orbit)
        features = df_orbit.iloc[:, 1:-5].values

        x = np.array(features, copy=True)
        x = x.reshape(-1, x.shape[1], 1)

        print_(
            f'testing classifier on orbit {orbit} ({df_orbit.iloc[0]["SPLIT"]})')
        pred = clf.predict(x)  # window vs step
        df.loc[df['ORBIT'] == orbit, 'LABEL_PRED'] = pred.argmax(axis=-1)

    return df


# %% Setup

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print_(e)

logs = sys.argv[1]
if not os.path.exists(logs):
    os.makedirs(logs)

fptr = open(f'{logs}/log_cnn.txt', 'w')


# %% Load data

drifts_orbits = load_drifts(f'{logs}/drifts.txt', 'w')
files = []
# TODO: load known orbits from different folder
for orb in drifts_orbits:
    files.append(f'data/orbits/df_{orb}.csv')

df = load_data(files, add_known_drifts=True)
df = select_features(df, 'data/features_gan.txt')
df['DRIFT'] = 1
df['SPLIT'] = 'train'

# Randomly select orbits for testing
len_train = 0
len_test = 0
for drift in np.unique(list(drifts_orbits.values())):
    all_orbits = [k for k, v in drifts_orbits.items() if v == drift]
    test_orbits = random.sample(test_orbits, len(test_orbits) // 5)
    train_orbits = [orb for orb in all_orbits if orb not in test_orbits]
    len_train += len(train_orbits)
    len_test += len(test_orbits)
    print_(f'train orbits for drift {drift}:')
    for orb in train_orbits:
        print_(f'{orb}')
        df.loc[df['ORBIT'] == orb, 'DRIFT'] = drift
    print_(f'test orbits for drift {drift}:')
    for orb in test_orbits:
        print_(f'{orb}')
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
all_pred = test_clfs(df.copy())
t2 = time.perf_counter()
print_(f'testing time is {t2 - t1:.2f} seconds')


# # %% Evaluation

# for n in orbits_all:
#     idx = orbits_all[n]
#     split = 'train'
#     if n in orbits_test:
#         split = 'test'
#     f1 = precision_recall_fscore_support(labels_all_true[idx[0]:idx[1]],
#                                          all_pred[idx[0]:idx[1]],
#                                          average=None,
#                                          labels=np.unique(labels_all_true[idx[0]:idx[1]]))[2]
#     for d, d_idx in drifts:
#         if d_idx == idx:
#             print_(
#                 f'{split} orbit {n} {idx} drift {d} f-score - {f1}')
#             break

# auc_value = accuracy_score(y_true=labels_train_true, y_pred=labels_train_pred)
# print_(f'accuracy value is {auc_value} for training dataset {dataset}')
# prf = precision_recall_fscore_support(
#     labels_train_true, labels_train_pred, average=None, labels=np.unique(labels_train_true))
# print_(f'precision: {prf[0]}')
# print_(f'recall: {prf[1]}')
# print_(f'f-score: {prf[2]}')
# print_(f'support: {prf[3]}')
# print_(
#     f'confusion matrix:\n{confusion_matrix(labels_train_true, labels_train_pred)}')

# auc_value = accuracy_score(y_true=labels_test_true, y_pred=labels_test_pred)
# print_(f'accuracy value is {auc_value} for testing dataset {dataset}')
# prf = precision_recall_fscore_support(
#     labels_test_true, labels_test_pred, average=None, labels=np.unique(labels_test_true))
# print_(f'precision: {prf[0]}')
# print_(f'recall: {prf[1]}')
# print_(f'f-score: {prf[2]}')
# print_(f'support: {prf[3]}')
# print_(
#     f'confusion matrix:\n{confusion_matrix(labels_test_true, labels_test_pred)}')
# print_(f'unique test true: {np.unique(labels_test_true)}')
# print_(f'unique test pred: {np.unique(labels_test_pred)}')


# # %% Plots

# if plots != '':
#     print_(f'plotting {plots}...')
#     if '0' in plots:
#         os.makedirs(f'../logs/{folder}/train-true')
#         plot_orbit(df_all, orbits_train, 'train-true')
#         print_(f'plotted train-true')
#     if '1' in plots:
#         os.makedirs(f'../logs/{folder}/train-pred')
#         plot_orbit(df_all, orbits_train, 'train-pred')
#         print_(f'plotted train-pred')
#     if '2' in plots:
#         os.makedirs(f'../logs/{folder}/test-true')
#         plot_orbit(df_all, orbits_test, 'test-true')
#         print_(f'plotted test-true')
#     if '3' in plots:
#         os.makedirs(f'../logs/{folder}/test-pred')
#         plot_orbit(df_all, orbits_test, 'test-pred')
#         print_(f'plotted test-pred')


# %% Close log file

if fptr is not None:
    fptr.close()
    fptr = None
