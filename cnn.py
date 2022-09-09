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
import wandb
from PIL import Image
from scipy.stats import entropy
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers, metrics
from wandb.keras import WandbCallback

from util import load_data, load_drifts, print_f, select_features

global fptr
"""File-like object for storing output as text."""


def print_(print_str, with_date=True):
    """
    Prints a value to a file-like object ``fptr`` and to sys.stdout.

    Args:
        print_str: Value to be printed.
        with_date: If ``True``, append current datetime before the value.
    """
    global fptr
    print_f(fptr, print_str, with_date)
    if with_date:
        print(f'{str(datetime.now())}: {print_str}', flush=True)
    else:
        print(print_str, flush=True)


def cnn(shape):
    """
    Initializes a CRNN classifier Keras model.

    Args:
        shape: Input shape for the first Conv1D layer of the model.

    Returns:
        model: Network model.
    """
    model = keras.Sequential()
    model.add(layers.Conv1D(64, 3, strides=1, activation='relu',
                            padding='same', input_shape=shape))
    model.add(layers.LSTM(64, return_sequences=True))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(5, activation='softmax'))

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy', metrics.sparse_categorical_accuracy])

    return model


def get_entropy(df):
    """
    Calculates entropy value of an orbit.

    Args:
        df: DataFrame with orbit data.

    Returns:
        e: Entropy value.
    """
    labels = df['LABEL'].tolist()
    _, counts = np.unique(labels, return_counts=True)
    e = entropy(counts)

    return e


def get_accuracy(y_true, y_pred):
    """
    Calculates accuracy value for each class.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        acc: Array of accuracy values for each class.
    """
    cm = confusion_matrix(y_true, y_pred)
    if len(cm) == 0 or cm.sum() == 0:
        return np.full(5, 0.0)
    acc = []
    for i in range(len(cm)):
        tp = cm[i][i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        acc.append((tp + tn) / cm.sum())
    acc = np.array(acc)

    return acc


def get_error_rate(y_true, y_pred):
    """
    Calculates error rate macro value and values for each class.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        er_macro: Macro error rate.
        er: Array of error rate values for each class.
    """
    cm = confusion_matrix(y_true, y_pred)
    if len(cm) == 0 or len(y_true) == 0:
        return 1.0, np.full(5, 1.0)
    er = []
    subs = []
    dels = []
    inss = []
    for i in range(len(cm)):
        tp = cm[i][i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        s = min(fn, fp)
        d = max(0, fn - fp)
        ins = max(0, fp - fn)
        e = (s + d + ins) / len(y_true)
        subs.append(s)
        dels.append(d)
        inss.append(ins)
        er.append(e)
    er_macro = (sum(subs) + sum(dels) + sum(inss)) / (len(cm) * len(y_true))
    er = np.array(er)

    return er_macro, er


def smooth(labels, window_size=120, window_size2=60):
    """
    Smoothes predicted labels to remove intermittency.

    Args:
        labels: Labels predicted by the classifier.
        window_size: Min size of an allowed crossing.
        window_size2: Max size of an allowed gap between crossings.

    Returns:
        labels: Smoothed labels.
    """
    for i in range(len(labels)-window_size2):
        window = labels[i:i+window_size2]
        if window[0] == window[-1]:
            if window[0] != 1 and window[0] != 3:
                labels[i:i+window_size2] = np.full(window_size2, window[0])
    for i in range(len(labels)-window_size):
        window = labels[i:i+window_size]
        if window[0] == window[-1]:
            if window[0] == 1 or window[0] == 3:
                labels[i:i+window_size] = np.full(window_size, window[0])

    return labels


def train_clf(df):
    """
    Creates and trains classifier on provided orbit data.

    Args:
        df: DataFrame with training data.

    Returns:
        labels_pred: Predicted labels for training data.
        clf: Trained classifier.
    """
    np.set_printoptions(precision=3)

    # Standardization
    df_features = df.iloc[:, 1:-5]
    len_features = len(df.iloc[:, 1:-5].columns)
    print_(f'features:\n{df.columns}')

    df.iloc[:, 1:-5] = (df_features - df_features.mean()) / df_features.std()
    print_(f'total size = {len(df.index)}')

    drifts = pd.unique(df['DRIFT']).tolist()
    print_(f'drifts: {drifts}')
    print_(f'========================================')

    clf = cnn((len_features, 1))
    print_(f'cnn for all drifts ({len_features} features):')
    clf.summary(print_fn=print_)

    df_train = df.loc[df['SPLIT'] == 'train']
    orbit_numbers = pd.unique(df_train['ORBIT']).tolist()
    print_(f'{len(orbit_numbers)} training orbits')
    print_(f'selected orbits for training: {orbit_numbers}')

    features = df_train.iloc[:, 1:-5].values
    labels = df_train['LABEL'].tolist()

    x_train = np.array(features, copy=True)
    x_train = x_train.reshape(-1, x_train.shape[1], 1)
    y_train = np.asarray(labels)
    classes = np.unique(y_train)

    weights = compute_class_weight(
        'balanced', classes=classes, y=y_train)
    print_(f'weights: {weights}')

    clf.fit(x=x_train, y=y_train,
            batch_size=16,
            epochs=10,
            callbacks=[WandbCallback()],
            class_weight={k: v for k,
                          v in enumerate(weights)},
            verbose=2)

    acc = clf.evaluate(x_train, y_train, verbose=2)
    print_(f'metric names: {clf.metrics_names}')
    print_(f'evaluation: {acc}')

    # Training evaluation
    labels_pred = clf.predict(x_train)
    labels_pred = labels_pred.argmax(axis=-1)
    labels_pred = smooth(labels_pred)
    df.loc[df['SPLIT'] == 'train', 'LABEL_PRED'] = labels_pred
    prf = precision_recall_fscore_support(
        y_true=y_train, y_pred=labels_pred, average=None, labels=classes)
    print_(f'training precision: {prf[0]}')
    print_(f'training recall: {prf[1]}')
    print_(f'training f-score: {prf[2]}')

    # Testing on validation set
    df_test = df.loc[df['SPLIT'] == 'valid']

    if len(df_test.index) > 0:
        orbit_numbers_test = pd.unique(df_test['ORBIT']).tolist()
        print_(f'{len(orbit_numbers_test)} validation orbits')

        features_test = df_test.iloc[:, 1:-5].values
        x_test = np.array(features_test, copy=True)
        x_test = x_test.reshape(-1, x_test.shape[1], 1)

        # Testing evaluation
        labels_pred_test = clf.predict(x_test)
        labels_pred_test = labels_pred_test.argmax(axis=-1)
        labels_pred_test = smooth(labels_pred_test)
        df.loc[df['SPLIT'] == 'valid', 'LABEL_PRED'] = labels_pred_test
        y_test = df_test['LABEL'].tolist()
        prf_test = precision_recall_fscore_support(
            y_true=y_test, y_pred=labels_pred_test, average=None, labels=np.unique(y_test))
        print_(f'validation precision: {prf_test[0]}')
        print_(f'validation recall: {prf_test[1]}')
        print_(f'validation f-score: {prf_test[2]}')

        labels_pred = df['LABEL_PRED']

    print_(f'========================================')

    return labels_pred, clf


# Test classifier
def test_clf(df, clf):
    """
    Tests classifier on provided orbit data.

    Args:
        df: DataFrame with testing orbit data.
        clf: Trained classifier.

    Returns:
        labels_pred: Predicted labels for testing data.
    """
    # Standardization
    df_features = df.iloc[:, 1:-5]
    print_(f'features:\n{df.columns}')

    df.iloc[:, 1:-5] = (df_features - df_features.mean()) / df_features.std()
    print_(f'total size = {len(df.index)}')

    features = df.iloc[:, 1:-5].values
    x = np.array(features, copy=True)
    x = x.reshape(-1, x.shape[1], 1)
    labels_pred = clf.predict(x)
    labels_pred = labels_pred.argmax(axis=-1)
    labels_pred = smooth(labels_pred)
    df['LABEL_PRED'] = labels_pred
    y_test = df['LABEL'].tolist()
    prf_test = precision_recall_fscore_support(
        y_true=y_test, y_pred=labels_pred, average=None, labels=np.unique(y_test))
    print_(f'testing precision: {prf_test[0]}')
    print_(f'testing recall: {prf_test[1]}')
    print_(f'testing f-score: {prf_test[2]}')

    labels_pred = df['LABEL_PRED']

    return labels_pred


# Plot all orbits with crossings
def plot_orbits(logs, dataset, df, orb_idx, max_orbits, test=False, pred=False, draw=[1, 3]):
    """
    Plots magnetic components from orbit data with labels.

    Args:
        logs: Directory for storing plot images.
        dataset: Number of a dataset sample.
        df: DataFrame with labelled orbit data.
        orb_idx: Initial indices at which orbit plots start and end.
        max_orbits: Max amount of training orbits for each class.
        test: ``True`` if testing orbits need to be plotted, ``False`` if training orbits.
        pred: ``True`` if predicted labels need to be plotted, ``False`` if true labels.
        draw: List of crossings that need to be plotted (0-4).

    Returns:
        orb_idx: Adjusted indices at which orbit plots start and end.
    """
    colours = {0: 'red', 1: 'green', 2: 'yellow', 3: 'blue', 4: 'purple'}
    title = ' labels'
    folder = 'train-'
    if test:
        folder = 'test-'
        df = df.loc[df['SPLIT'] == 'test']
    else:
        df = df.loc[df['SPLIT'] != 'test']

    label_col = 'LABEL'
    if pred:
        label_col = 'LABEL_PRED'
        title = 'predicted' + title
        folder += 'pred'
    else:
        title = 'true' + title
        folder += 'true'

    if not os.path.exists(f'{logs}/plots_set{dataset}_{max_orbits}/{folder}'):
        os.makedirs(f'{logs}/plots_set{dataset}_{max_orbits}/{folder}')

    orbits = pd.unique(df['ORBIT']).tolist()
    for orbit in orbits:

        df_orbit = df.loc[df['ORBIT'] == orbit]
        if orbit in orb_idx:
            start, end = orb_idx[orbit]
        else:
            idx = df_orbit.index[df_orbit[label_col] == 1]
            start = max(idx[0]-1500, df_orbit.index[0])
            end = min(idx[-1]+1500, df_orbit.index[-1])
            orb_idx[orbit] = (start, end)
        df_orbit = df_orbit.loc[start:end]

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

        ann = {1: 'SK', 3: 'MP'}
        for i in draw:
            df_label = df_orbit.loc[df_orbit[label_col] == i]
            if len(df_label) > 0:
                fig.add_annotation(x=df_label.iloc[0]['DATE'], y=450,
                                   text=ann[i],
                                   showarrow=False,
                                   yshift=10)
                fig.add_annotation(x=df_label.iloc[-1]['DATE'], y=450,
                                   text=ann[i],
                                   showarrow=False,
                                   yshift=10)
            for _, row in df_label.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row['DATE'], row['DATE']],
                    y=[-450, 450],
                    mode='lines',
                    line_color=colours[i],
                    opacity=0.005,
                    showlegend=False
                ))

        fig.update_layout(
            {'title': f'Orbit {orbit} (drift {df_orbit.iloc[0]["DRIFT"]}) {title}'})
        fig.write_image(
            f'{logs}/plots_set{dataset}_{max_orbits}/{folder}/fig{orbit}_drift{df_orbit.iloc[0]["DRIFT"]}.png')

        if (orbits.index(orbit) + 1) % 10 == 0:
            print_(
                f'orbit {orbits.index(orbit) + 1}/{len(orbits)} (fig{orbit}_drift{df_orbit.iloc[0]["DRIFT"]}.png)')

    return orb_idx


def merge_plots(folder, split):
    """
    Merges plots with true and predicted crossings into one image.

    Args:
        folder: Directory for storing merged images.
        split: ``train`` or ``test``.
    """
    img_files = glob.glob(f'{folder}/{split}-true/*.png')

    if not os.path.exists(f'{folder}/{split}-all'):
        os.makedirs(f'{folder}/{split}-all')

    for img in img_files:
        fig_true = Image.open(img)
        fig_pred = Image.open(img.replace('true', 'pred'))
        fig_all = Image.new(
            'RGB', (fig_true.size[0], 2*fig_true.size[1] - 40), (255, 255, 255))
        fig_all.paste(fig_true, (0, 0))
        fig_all.paste(fig_pred, (0, fig_true.size[1] - 40))
        fig_all.save(img.replace('true', 'all'), 'PNG')


# GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print_(e)

# Command line arguments
logs = sys.argv[1]
dataset = int(sys.argv[2])
plots = sys.argv[3]
max_orbits = int(sys.argv[4])
if not os.path.exists(logs):
    os.makedirs(logs)

fptr = open(f'{logs}/log_cnn_set{dataset}_{max_orbits}.txt', 'w')
print_(f'dataset: {dataset}')

wandb.init(project="cnn", entity="irodionr", config={
    "filters": 64,
    "kernel_size": 2,
    "units_lstm": 64,
    "units_dense": 16,
    "batch_size": 128,
    "epochs": 10,
    "max_orbits": max_orbits,
    "learning_rate": 0.001
})

# Loading data
drift_orbits = load_drifts(f'data/drifts_set{dataset}.txt')
_, _, known_drifts = next(os.walk('data/drifts/'))
known_drift_count = len(known_drifts)
files = []
cur_orbit = 0
for orb in drift_orbits:
    if cur_orbit < known_drift_count:
        files.append(f'data/drifts/df_{orb}.csv')
        print_(f'data/drifts/df_{orb}.csv', with_date=False)
    else:
        files.append(f'data/orbits/df_{orb}.csv')
        print_(f'data/orbits/df_{orb}.csv', with_date=False)
    cur_orbit += 1

df = load_data(files)
df = select_features(df, 'data/features_cnn.txt')

df['DRIFT'] = 1
df['LABEL_PRED'] = 0
df['SPLIT'] = 'test'
print_(f'selected data:\n{df.columns}')

# Assigning drifts to orbits
for drift in pd.unique(list(drift_orbits.values())).tolist():
    all_orbits = [k for k, v in drift_orbits.items() if v == drift]
    print_(f'{len(all_orbits)} orbits with drift {drift}')
    for orb in all_orbits:
        df.loc[df['ORBIT'] == orb, 'DRIFT'] = drift

# Select orbits for testing
np.set_printoptions(precision=3)
total_train = []
total_valid = []
total_test = []
for drift in pd.unique(df['DRIFT']).tolist():
    df_drift = df.loc[df['DRIFT'] == drift]
    list_orbits = []
    list_all = pd.unique(df_drift['ORBIT']).tolist()
    for orbit in list_all:
        list_orbits.append(df_drift.loc[df_drift['ORBIT'] == orbit])

    test_count = max(len(list_orbits) // 5, 1)  # 20% of orbits or 1
    valid_count = len(list_orbits) - test_count

    # Pick orbits for validation set
    list_valid_orbits = random.sample(list_orbits, valid_count)
    list_valid = []
    for orb in list_valid_orbits:
        list_valid.append(orb.iloc[0]["ORBIT"])
        df.loc[df['ORBIT'] == orb.iloc[0]['ORBIT'], 'SPLIT'] = 'valid'

    list_test = [orb for orb in list_all if orb not in list_valid]
    list_test_orbits = []
    for orbit in list_test:
        list_test_orbits.append(df_drift.loc[df_drift['ORBIT'] == orbit])

    # List of orbits in a drift with descending entropy
    list_valid_orbits.sort(key=get_entropy, reverse=True)
    train_count = min(max_orbits, len(list_valid_orbits))
    list_train_orbits = list_valid_orbits[:train_count]
    list_valid_orbits = list_valid_orbits[train_count:]

    list_train = []
    for orb in list_train_orbits:
        list_train.append(orb.iloc[0]["ORBIT"])
        df.loc[df['ORBIT'] == orb.iloc[0]['ORBIT'], 'SPLIT'] = 'train'

    list_valid = []
    for orb in list_valid_orbits:
        list_valid.append(orb.iloc[0]["ORBIT"])

    total_train += list_train
    print_(f'drift {drift} training orbits ({len(list_train)}): {list_train}')
    total_valid += list_valid
    total_test += list_test
    print_(f'drift {drift} testing orbits ({len(list_test)}): {list_test}')

if not total_train:
    total_train.append(total_valid[0])
    total_valid = total_valid[1:]
    df.loc[df['ORBIT'] == total_train[0], 'SPLIT'] = 'train'
    print_(f'add 1 training orbit')

wandb.log({"training_orbits": len(total_train)})
print_(f'total training orbits: {len(total_train)}')
print_(f'{total_train}')
print_(f'total testing orbits: {len(total_test)}')
print_(f'{total_test}')

print_(f'========== TRAINING ==========')
t1 = time.perf_counter()
train_preds, clfs = train_clf(df.loc[df['SPLIT'] != 'test'].copy())
t2 = time.perf_counter()
print_(f'training time is {t2 - t1:.2f} seconds')

df.loc[df['SPLIT'] != 'test', 'LABEL_PRED'] = train_preds

print_(f'========== TESTING ==========')
t1 = time.perf_counter()
test_preds = test_clf(df.loc[df['SPLIT'] == 'test'].copy(), clfs)
t2 = time.perf_counter()
print_(f'testing time is {t2 - t1:.2f} seconds')

df.loc[df['SPLIT'] == 'test', 'LABEL_PRED'] = test_preds


print_(f'========== EVALUATION ==========')
drifts = pd.unique(df['DRIFT']).tolist()
print_(f'drifts: {drifts}')

for drift in drifts:

    df_drift = df.loc[df['DRIFT'] == drift]

    labels = df_drift.loc[df_drift['SPLIT'] == 'test', 'LABEL'].tolist()
    labels_pred = df_drift.loc[df_drift['SPLIT']
                               == 'test', 'LABEL_PRED'].tolist()
    classes = np.unique(labels)

    prf = precision_recall_fscore_support(
        y_true=labels, y_pred=labels_pred, average=None, labels=classes)
    print_(
        f'drift {drift} testing f-score: {prf[2]}, recall: {prf[1]}, precision: {prf[0]}')

    orbit_numbers = pd.unique(df_drift['ORBIT']).tolist()
    print_(f'{len(orbit_numbers)} orbits with drift {drift}')
    print_(f'{orbit_numbers}')

    for orbit in orbit_numbers:

        df_orbit = df_drift.loc[df_drift['ORBIT'] == orbit]
        labels = df_orbit['LABEL'].tolist()
        labels_pred = df_orbit['LABEL_PRED'].tolist()
        classes = np.unique(labels)

        prf = precision_recall_fscore_support(
            y_true=labels, y_pred=labels_pred, average=None, labels=classes)
        print_(
            f'{df_orbit.iloc[0]["SPLIT"]} orbit {orbit} f-score: {prf[2]}, recall: {prf[1]}, precision: {prf[0]}')

labels_train_true = df.loc[df['SPLIT'] == 'train', 'LABEL'].tolist()
labels_train_pred = df.loc[df['SPLIT'] == 'train', 'LABEL_PRED'].tolist()
labels_valid_true = df.loc[df['SPLIT'] == 'valid', 'LABEL'].tolist()
labels_valid_pred = df.loc[df['SPLIT'] == 'valid', 'LABEL_PRED'].tolist()
labels_test_true = df.loc[df['SPLIT'] == 'test', 'LABEL'].tolist()
labels_test_pred = df.loc[df['SPLIT'] == 'test', 'LABEL_PRED'].tolist()

auc_value = accuracy_score(y_true=labels_train_true, y_pred=labels_train_pred)
print_(f'accuracy value is {auc_value} for training dataset')
prf = precision_recall_fscore_support(
    labels_train_true, labels_train_pred, average=None, labels=np.unique(labels_train_true))
er_macro, er = get_error_rate(labels_train_true, labels_train_pred)
acc = get_accuracy(labels_train_true, labels_train_pred)
print_(f'accuracy: {acc}')
print_(f'macro error rate: {er_macro}')
print_(f'error rate: {er}')
print_(f'precision: {prf[0]}')
print_(f'recall: {prf[1]}')
print_(f'f-score: {prf[2]}')
print_(f'support: {prf[3]}')
print_(
    f'confusion matrix:\n{confusion_matrix(labels_train_true, labels_train_pred)}')

auc_value = accuracy_score(y_true=labels_valid_true, y_pred=labels_valid_pred)
print_(f'accuracy value is {auc_value} for validation dataset')
prf = precision_recall_fscore_support(
    labels_valid_true, labels_valid_pred, average=None, labels=np.unique(labels_valid_true))
er_macro, er = get_error_rate(labels_valid_true, labels_valid_pred)
acc = get_accuracy(labels_valid_true, labels_valid_pred)
print_(f'accuracy: {acc}')
print_(f'macro error rate: {er_macro}')
print_(f'error rate: {er}')
print_(f'precision: {prf[0]}')
print_(f'recall: {prf[1]}')
print_(f'f-score: {prf[2]}')
print_(f'support: {prf[3]}')
print_(
    f'confusion matrix:\n{confusion_matrix(labels_valid_true, labels_valid_pred)}')

auc_value = accuracy_score(y_true=labels_test_true, y_pred=labels_test_pred)
print_(f'accuracy value is {auc_value} for testing dataset')
wandb.log({"macro accuracy": auc_value})
prf = precision_recall_fscore_support(
    labels_test_true, labels_test_pred, average=None, labels=np.unique(labels_test_true))
er_macro, er = get_error_rate(labels_test_true, labels_test_pred)
acc = get_accuracy(labels_test_true, labels_test_pred)
print_(f'accuracy: {acc}')
print_(f'macro error rate: {er_macro}')
print_(f'error rate: {er}')
print_(f'precision: {prf[0]}')
print_(f'recall: {prf[1]}')
print_(f'f-score: {prf[2]}')
print_(f'support: {prf[3]}')
print_(
    f'confusion matrix:\n{confusion_matrix(labels_test_true, labels_test_pred)}')
wandb.log({"SK accuracy": acc[1]})
wandb.log({"MP accuracy": acc[3]})
wandb.log({"SK error rate": er[1]})
wandb.log({"MP error rate": er[3]})
wandb.log({"SK precision": prf[0][1]})
wandb.log({"MP precision": prf[0][3]})
wandb.log({"SK recall": prf[1][1]})
wandb.log({"MP recall": prf[1][3]})
wandb.log({"SK f-score": prf[2][1]})
wandb.log({"MP f-score": prf[2][3]})

# Plotting
if plots != '5':
    df['B_tot'] = (df['BX_MSO']**2 + df['BY_MSO']**2 + df['BZ_MSO']**2)**0.5
    orb_idx = {}
    print_(f'plotting {plots}...')
    if '0' in plots:
        plot_orbits(logs, dataset, df.copy(), orb_idx,
                    max_orbits, test=False, pred=False)
        print_(f'plotted train-true')
    if '1' in plots:
        plot_orbits(logs, dataset, df.copy(), orb_idx,
                    max_orbits, test=False, pred=True)
        print_(f'plotted train-pred')
        merge_plots(f'{logs}/plots_set{dataset}_{max_orbits}', 'train')
        print_(f'merged train plots')
    if '2' in plots:
        plot_orbits(logs, dataset, df.copy(), orb_idx,
                    max_orbits, test=True, pred=False)
        print_(f'plotted test-true')
    if '3' in plots:
        plot_orbits(logs, dataset, df.copy(), orb_idx,
                    max_orbits, test=True, pred=True)
        print_(f'plotted test-pred')
        merge_plots(f'{logs}/plots_set{dataset}_{max_orbits}', 'test')
        print_(f'merged test plots')

if fptr is not None:
    fptr.close()
    fptr = None
