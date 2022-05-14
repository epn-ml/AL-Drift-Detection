# %% imports

import glob
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers
from torch import nn
from torch.autograd import Variable
from torch.nn import Linear, Module, ReLU, Sequential
from torch.optim import Adadelta
from torch.utils.data import DataLoader

from util import load_data, print_f, select_features

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


# Train classifiers based on drifts
def train_clfs(features, labels, drifts):

    clfs = {}
    classes = np.unique(labels)
    # Get drift labels from [(1, ()), ...]
    drift_labels = list(map(list, zip(*drifts)))[0]
    # Split drifts into 2 groups
    drift_threshold = len(np.unique(drift_labels)) // 2
    print_(
        f'group 1 = 1..{drift_threshold}, group 2 = {drift_threshold+1}..{max(drift_labels)}')
    weights = compute_class_weight(
        'balanced', classes=classes, y=labels)
    print_(f'weights = {weights}')

    for d in drifts:

        drift_num = d[0]
        drift_idx = d[1]
        group = 1
        if drift_num > drift_threshold:
            group = 2

        if drift_idx[0] < len(features):

            bound = drift_idx[1]

            if bound > len(features):
                bound = len(features)
                print_(
                    f'index {drift_idx[1]} is outside of training orbits, set to {len(features)}')

            x = np.array(features[drift_idx[0]:bound, :], copy=True)
            x = x.reshape(-1, x.shape[1], 1)
            y = np.asarray(labels[drift_idx[0]:bound])

            # Train group classifiers on drifts of that group
            if not group in clfs:
                clfs[group] = cnn(x.shape[1:])
                print_(f'create new classifier for drift group {group}')

            print_(
                f'training classifier {group} on drift {drift_num} - {(drift_idx[0], bound)}...')
            clfs[group].fit(x=x, y=y,
                            batch_size=64,
                            epochs=20,
                            class_weight={k: v for k,
                                          v in enumerate(weights)},
                            verbose=0)

        else:
            print_(f'{drift_idx} is outside of training orbits, ignoring')

    print_(f'trained classifiers for drifts groups - {list(clfs.keys())}')

    return clfs


# Test classifiers
def test_clfs(features, drifts, clfs):

    labels = list(range(len(features)))
    drift_labels = list(map(list, zip(*drifts)))[0]
    drift_threshold = len(np.unique(drift_labels)) // 2

    for d in drifts:

        drift_num = d[0]
        drift_idx = d[1]

        group = 1
        if drift_num > drift_threshold:
            group = 2

        if group in clfs:
            g = group
        else:
            g = min(clfs.keys(), key=lambda x: abs(x-group))
            print_(
                f'no classifier for drift group {group}, switching to {g}')

        x = np.array(features[drift_idx[0]:drift_idx[1]], copy=True)
        x = x.reshape(-1, x.shape[1], 1)

        print_(
            f'testing classifier for drift group {g}, drift {drift_num} - {drift_idx}...')
        pred = clfs[g].predict(x)  # window vs step
        labels[drift_idx[0]:drift_idx[1]] = pred.argmax(axis=-1)

    return labels


# %% Setup

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print_(e)


# %% Training classifiers

t1 = time.perf_counter()
clfs = train_clfs(features=features_train,
                  labels=labels_train_true, drifts=drifts)
t2 = time.perf_counter()
print_(f'training time is {t2 - t1:.2f} seconds')


# %% Testing classifiers

t1 = time.perf_counter()
all_pred = test_clfs(features_all, drifts, clfs)
t2 = time.perf_counter()
print_(f'testing time is {t2 - t1:.2f} seconds')

df_all['LABEL_PRED'] = all_pred
labels_train_pred = all_pred[:len(features_train)]
labels_test_pred = all_pred[-len(features_test):]


# %% Evaluation

for n in orbits_all:
    idx = orbits_all[n]
    split = 'train'
    if n in orbits_test:
        split = 'test'
    f1 = precision_recall_fscore_support(labels_all_true[idx[0]:idx[1]],
                                         all_pred[idx[0]:idx[1]],
                                         average=None,
                                         labels=np.unique(labels_all_true[idx[0]:idx[1]]))[2]
    for d, d_idx in drifts:
        if d_idx == idx:
            print_(
                f'{split} orbit {n} {idx} drift {d} f-score - {f1}')
            break

auc_value = accuracy_score(y_true=labels_train_true, y_pred=labels_train_pred)
print_(f'accuracy value is {auc_value} for training dataset {dataset}')
prf = precision_recall_fscore_support(
    labels_train_true, labels_train_pred, average=None, labels=np.unique(labels_train_true))
print_(f'precision: {prf[0]}')
print_(f'recall: {prf[1]}')
print_(f'f-score: {prf[2]}')
print_(f'support: {prf[3]}')
print_(
    f'confusion matrix:\n{confusion_matrix(labels_train_true, labels_train_pred)}')

auc_value = accuracy_score(y_true=labels_test_true, y_pred=labels_test_pred)
print_(f'accuracy value is {auc_value} for testing dataset {dataset}')
prf = precision_recall_fscore_support(
    labels_test_true, labels_test_pred, average=None, labels=np.unique(labels_test_true))
print_(f'precision: {prf[0]}')
print_(f'recall: {prf[1]}')
print_(f'f-score: {prf[2]}')
print_(f'support: {prf[3]}')
print_(
    f'confusion matrix:\n{confusion_matrix(labels_test_true, labels_test_pred)}')
print_(f'unique test true: {np.unique(labels_test_true)}')
print_(f'unique test pred: {np.unique(labels_test_pred)}')


# %% Plots

if plots != '':
    print_(f'plotting {plots}...')
    if '0' in plots:
        os.makedirs(f'../logs/{folder}/train-true')
        plot_orbit(df_all, orbits_train, 'train-true')
        print_(f'plotted train-true')
    if '1' in plots:
        os.makedirs(f'../logs/{folder}/train-pred')
        plot_orbit(df_all, orbits_train, 'train-pred')
        print_(f'plotted train-pred')
    if '2' in plots:
        os.makedirs(f'../logs/{folder}/test-true')
        plot_orbit(df_all, orbits_test, 'test-true')
        print_(f'plotted test-true')
    if '3' in plots:
        os.makedirs(f'../logs/{folder}/test-pred')
        plot_orbit(df_all, orbits_test, 'test-pred')
        print_(f'plotted test-pred')


# %% Close log file

if fptr is not None:
    fptr.close()
    fptr = None
