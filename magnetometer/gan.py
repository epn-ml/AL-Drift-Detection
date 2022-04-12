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

global seq_len
global fptr
global dataset
global folder
global plot_format


# %% functions

# Function to write the log to disk also
def print_(print_str):
    global fptr
    global dataset
    if fptr is None:
        os.makedirs(f'../logs/{folder}')
        name = f'../logs/{folder}/log.txt'
        fptr = open(name, "w")

    fptr.write(str(datetime.now()) + ": " + str(print_str) + "\n")
    # print(str(datetime.now()) + ": " + str(print_str))
    fptr.flush()
    os.fsync(fptr.fileno())


class Generator(Module):
    def __init__(self, inp, out, sequence_length=2, num_layers=3):
        super(Generator, self).__init__()
        self.net = Sequential(
            Linear(inp*sequence_length, 128),
            Linear(128, 4096), ReLU(inplace=True),
            # Linear(128, 256), ReLU(inplace=True),
            # Linear(256, 256), ReLU(inplace=True),
            # Linear(256, 1024), Dropout(inplace=True),
            Linear(4096, inp)
        )

    def forward(self, x_):
        x_r = x_.reshape(x_.shape[0], x_.shape[1] * x_.shape[2])
        output = self.net(x_r)

        # output = output.reshape(output.shape[0], output.shape[1] * output.shape[2])
        return output

    def move(self, device):
        pass


class Discriminator(Module):
    def __init__(self, inp, final_layer_incoming_connections=512):
        super(Discriminator, self).__init__()
        self.input_connections = inp
        self.neuron_count = 2
        self.incoming_connections = final_layer_incoming_connections

        self.net = self.create_network()

        self.neurons = Linear(
            final_layer_incoming_connections, self.neuron_count)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_):
        result = self.net(x_)
        result = self.neurons(result)
        result = self.softmax(result)
        return result

    def update(self):
        # self.reset_layers()
        self.neuron_count += 1
        layer = Linear(self.incoming_connections, self.neuron_count)
        self.neurons = layer
        return

    def reset_top_layer(self):
        # self.reset_layers()
        layer = Linear(self.incoming_connections, self.neuron_count)
        self.neurons = layer
        return

    def reset_layers(self):
        self.net = self.create_network()

    def create_network(self):
        net = Sequential(
            Linear(self.input_connections, 1024),
            Linear(1024, 1024), ReLU(inplace=True),
            Linear(1024, self.incoming_connections),
            nn.Sigmoid())

        return net


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


def fit_and_predict(clf, features, labels, classes, weights):
    predicted = np.empty(shape=len(labels))
    predicted[0] = clf.predict([features[0]])
    clf.reset()
    clf.partial_fit([features[0]], [labels[0]],
                    classes=classes, sample_weight=[weights[labels[0]]])
    for idx in range(1, len(labels)):
        predicted[idx] = clf.predict([features[idx]])
        clf.partial_fit([features[idx]], [labels[idx]],
                        classes=classes, sample_weight=[weights[labels[idx]]])

    return predicted, clf


def fit(clf, features, labels, classes, weights):
    for idx in range(0, len(labels)):
        clf.partial_fit([features[idx]], [labels[idx]],
                        classes=classes, sample_weight=[weights[labels[idx]]])

    return clf


def predict_and_partial_fit(clf, features, labels, classes, weights):
    predicted = np.empty(shape=len(labels))
    for idx in range(0, len(labels)):
        predicted[idx] = clf.predict([features[idx]])
        clf.partial_fit([features[idx]], [labels[idx]],
                        classes=classes, sample_weight=[weights[labels[idx]]])

    return predicted, clf


def predict(clf, features):
    predicted = np.empty(shape=len(features))
    for idx in range(0, len(features)):
        predicted[idx] = clf.predict([features[idx]])

    return predicted


def collate(batch):
    """
    Function for collating the batch to be used by the data loader. This function does not handle labels
    :param batch:
    :return:
    """
    # Stack each tensor variable
    x = torch.stack([torch.tensor(x[:-1]) for x in batch])
    y = torch.Tensor([x[-1] for x in batch]).to(torch.long)
    # Return features and labels
    return x, y


def collate_generator(batch):
    """
    Function for collating the batch to be used by the data loader. This function does handle labels
    :param batch:
    :return:
    """
    global seq_len

    # Stack each tensor variable
    feature_length = int(len(batch[0]) / (seq_len + 1))
    # The last feature length corresponds to the feature we want to predict and
    # the last value is the label of the drift class
    x = torch.stack([torch.Tensor(np.reshape(x[:-feature_length-1], newshape=(seq_len, feature_length)))
                     for x in batch])
    y = torch.stack([torch.tensor(x[-feature_length-1:-1]) for x in batch])
    labels = torch.stack([torch.tensor(x[-1]) for x in batch])

    # Return features and targets
    return x.to(torch.double), y, labels


def train_discriminator(real_data, fake_data, discriminator, generator, optimizer, loss_fn,
                        generator_labels, device):
    # for idx in range(steps):
    for features, labels in real_data:
        # Set the gradients as zero
        discriminator.zero_grad()
        optimizer.zero_grad()

        # Get the loss when the real data is compared to ones
        features = features.to(device).to(torch.float)
        labels = labels.to(device)
        # features = features.to(torch.float)

        # Get the output for the real features
        output_discriminator = discriminator(features)

        # The real data is without any concept drift. Evaluate loss against zeros
        real_data_loss = loss_fn(output_discriminator, labels)

        # Get the output from the generator for the generated data compared to ones which is drifted data
        generator_input = None
        for input_sequence, _, _ in fake_data:
            generator_input = input_sequence.to(device).to(torch.float)
            break
        generated_output = generator(generator_input)  # .double().to(device))

        generated_output_discriminator = discriminator(generated_output)

        # Here instead of ones it should be the label of the drift category
        generated_data_loss = loss_fn(
            generated_output_discriminator, generator_labels)

        # Add the loss and compute back prop
        total_iter_loss = generated_data_loss + real_data_loss
        total_iter_loss.backward()

        # Update parameters
        optimizer.step()

    return discriminator


def train_generator(data_loader, discriminator, generator, optimizer, loss_fn, loss_mse, steps, device):
    epoch_loss = 0
    for idx in range(steps):

        optimizer.zero_grad()
        generator.zero_grad()

        generated_input = target = labels = None
        for generator_input, target, l in data_loader:
            generated_input = generator_input.to(torch.float).to(device)
            target = target.to(torch.float).to(device)
            labels = l.to(torch.long).to(device)
            # target = target.reshape((target.shape[0], target.shape[2]))
            break

        # Generating data for input to generator
        generated_output = generator(generated_input)

        # Compute loss based on whether discriminator can discriminate real data from generated data
        generated_training_discriminator_output = discriminator(
            generated_output)

        # Compute loss based on ideal target values
        loss_generated = loss_fn(
            generated_training_discriminator_output, labels)

        loss_lstm = loss_mse(generated_output, target)

        total_generator_loss = loss_generated + loss_lstm

        # Back prop and parameter update
        total_generator_loss.backward()
        optimizer.step()
        epoch_loss += total_generator_loss.item()

    return generator


def concatenate_features(data, sequence_len=2, has_label=True):
    if has_label is True:
        modified_data = data[:, :-1]
    else:
        modified_data = data

    idx = sequence_len
    modified_data = np.vstack(
        (np.zeros((sequence_len - 1, len(modified_data[idx]))), modified_data))
    output = np.hstack(
        (modified_data[idx - sequence_len:idx + 1, :].flatten(), data[idx-sequence_len][-1]))
    idx += 1

    while idx < len(modified_data)-1:
        if idx % 10000 == 0:
            print_(f'{idx}/{len(modified_data)-1} concatenating features...')
        output = np.vstack((output, np.hstack((modified_data[idx - sequence_len:idx + 1, :].flatten(),
                                               data[idx-sequence_len][-1]))))
        idx += 1

    # The last value
    output = np.vstack((output, np.hstack(
        (modified_data[idx - sequence_len:, :].flatten(), data[-1][-1]))))
    output = np.vstack((output, np.hstack((modified_data[idx - sequence_len:idx, :].flatten(),
                                           modified_data[sequence_len - 1],
                                           data[0][-1]))))
    return output


# Select features according to drift indices and append drift labeles
def create_training_dataset(dataset, indices, drift_labels):

    # print_(f'creating training dataset...')

    # If there is a periodicity, we switch all previous drifts to the same label
    modified_drift_labels = [x for x in drift_labels]
    if drift_labels[-1] != 0:
        modified_drift_labels = []
        for label in drift_labels:
            if label == drift_labels[-1]:
                modified_drift_labels.append(0)  # The current label
            elif label > drift_labels[-1]:
                # Decrease all labels that are greater than this
                modified_drift_labels.append(label-1)
            else:
                modified_drift_labels.append(label)

    # print_(f'modified dataset labels = {modified_drift_labels}')

    training_dataset = np.hstack((dataset[indices[0][0]:indices[0][1]],
                                  np.ones((indices[0][1]-indices[0][0], 1)) * modified_drift_labels[0]))
    for idx in range(1, len(modified_drift_labels)):
        training_dataset = np.vstack((training_dataset, np.hstack((dataset[indices[idx][0]:indices[idx][1]],
                                      np.ones((indices[idx][1]-indices[idx][0], 1)) * modified_drift_labels[idx]))))

    return training_dataset


def equalize_classes(features, max_count=100):
    modified_dataset = None

    labels = features[:, -1]
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_count = min(min(counts), max_count)

    if min_count == max(counts) == max_count:
        return features

    for label, count in zip(unique_labels, counts):
        indices = np.where(features[:, -1] == label)[0]
        chosen_indices = np.random.choice(indices, min_count)
        if modified_dataset is None:
            modified_dataset = features[chosen_indices, :]
            continue
        modified_dataset = np.vstack(
            (modified_dataset, features[chosen_indices, :]))
    return modified_dataset


def concat_feature(data, idx, sequence_len=2):
    if idx < len(data) - sequence_len - 1:
        return data[idx:idx + sequence_len + 1, :].flatten()
    if idx == len(data) - sequence_len - 1:
        return data[idx:, :].flatten()
    return np.hstack((data[idx:idx + sequence_len, :].flatten(), data[sequence_length - 1]))


def equalize_and_concatenate(features, max_count=100, sequence_len=2):
    modified_features = features[:, :-1]
    modified_features = np.vstack(
        (np.zeros((sequence_len - 1, len(modified_features[sequence_len]))), modified_features))

    labels = features[:, -1]
    labels[-1] = features[0][-1]

    unique_labels, counts = np.unique(labels, return_counts=True)
    min_count = min(min(counts), max_count)  # change max_count?

    if min_count == max(counts) == max_count:
        print_(f'counts = {counts} (min_count = {min_count})')
        return concatenate_features(features, sequence_len=sequence_len)

    output = None

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        chosen_indices = np.random.choice(indices, min_count)
        for idx in chosen_indices:
            if output is None:
                output = np.hstack(
                    (concat_feature(modified_features, idx, sequence_len=sequence_len), labels[idx]))
                continue
            output = np.vstack((output, np.hstack((concat_feature(
                modified_features, idx, sequence_len=sequence_len), labels[idx]))))

    return output


def train_gan(features, device, discriminator, generator, epochs=100, steps_generator=100, weight_decay=0.0005,
              max_label=1, generator_batch_size=1, seed=0, batch_size=8, lr=0.0001, momentum=0.9, equalize=True,
              sequence_length=2):

    # Set the seed for torch and numpy
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)
    np.random.seed(seed)

    # Losses for the generator and discriminator
    loss_mse_generator = nn.MSELoss()
    loss_generator = nn.CrossEntropyLoss()
    loss_discriminator = nn.CrossEntropyLoss()

    # Create the optimizers for the models
    optimizer_generator = Adadelta(
        generator.parameters(), weight_decay=weight_decay)
    optimizer_discriminator = Adadelta(
        discriminator.parameters(), weight_decay=weight_decay)

    # Label vectors
    ones = Variable(torch.ones(generator_batch_size)).to(torch.long).to(device)

    # print_(f'equalizing and concatenating data...')
    if equalize:
        # equalize and concatenate at the same time
        concatenated_data = equalize_and_concatenate(
            features, sequence_len=sequence_length)
        features = equalize_classes(features)
        # concatenated_data = equalize_classes(concatenated_data)
    else:
        # This data contains the current vector and next vector
        concatenated_data = concatenate_features(
            features, sequence_len=sequence_length)

    # Define the data loader for training
    real_data = DataLoader(features, batch_size=batch_size,
                           shuffle=True, collate_fn=collate)
    generator_data = DataLoader(concatenated_data, batch_size=generator_batch_size, shuffle=False,
                                collate_fn=collate_generator)

    # This is the label for new drifts (any input other than the currently learned distributions)
    generator_label = ones * max_label

    print_(f'training GAN...')
    for epochs_trained in range(epochs):
        discriminator = train_discriminator(real_data=real_data, fake_data=generator_data, discriminator=discriminator,
                                            generator=generator, optimizer=optimizer_discriminator,
                                            loss_fn=loss_discriminator, generator_labels=generator_label, device=device)

        generator = train_generator(data_loader=generator_data, discriminator=discriminator, generator=generator,
                                    optimizer=optimizer_generator, loss_fn=loss_generator, loss_mse=loss_mse_generator,
                                    steps=steps_generator, device=device)

    return generator, discriminator


def detect_drifts(features, orbits, dates, device, epochs=100, steps_generator=100, equalize=True, test_batch_size=4,
                  seed=0, batch_size=8, lr=0.001, momentum=0.9, weight_decay=0.0005,
                  generator_batch_size=1, sequence_length=2, repeat_factor=4):

    random.seed(seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.set_deterministic(True)

    current_batch_size = batch_size
    drifts_detected = [0]
    generator_label = 1

    # Create the Generator and Discriminator objects
    generator = Generator(
        inp=features.shape[1], out=features.shape[1], sequence_length=sequence_length)
    discriminator = Discriminator(
        inp=features.shape[1], final_layer_incoming_connections=512)

    generator.move(device=device)

    # Set the models to the device
    generator = generator.to(device=device)
    discriminator = discriminator.to(device=device)

    orbit_numbers = list(orbits.keys())
    orbits_idx = list(orbits.values())
    drift_indices = [orbits_idx[0]]
    cur_orbit = 1
    drift_labels = []

    temp_label = [0]
    initial_epochs = epochs * 2

    # Create training dataset
    print_(f'training dataset indices = {drift_indices}')
    print_(f'training dataset labels  = {[1]}')
    training_dataset = create_training_dataset(
        dataset=features, indices=drift_indices, drift_labels=[1])

    generator, discriminator = train_gan(features=training_dataset, device=device, discriminator=discriminator,
                                         generator=generator, epochs=initial_epochs, steps_generator=steps_generator,
                                         seed=seed, batch_size=batch_size, lr=lr, momentum=momentum, equalize=equalize,
                                         max_label=generator_label, generator_batch_size=generator_batch_size,
                                         weight_decay=weight_decay, sequence_length=sequence_length)

    index = orbits_idx[cur_orbit][0]

    generator.eval()
    discriminator.eval()

    no_drifts = index
    max_idx_prev = np.array([0, 0, 0, 0])

    print_(
        f'starting drift detection from index = {index} (orbit {orbit_numbers[cur_orbit]} - {orbits_idx[cur_orbit]} - {dates[index]})')
    print_('===========================')

    # while index + training_window_size < len(features):
    while index < orbits_idx[-1][-1]:

        data = features[index:index + test_batch_size]
        result = discriminator(torch.Tensor(data).to(torch.float).to(device))
        prob, max_idx = torch.max(result, dim=1)
        max_idx = max_idx.cpu().detach().numpy()

        drift_found = False

        # 1st condition is always false? (max_idx[1:] != ...)
        if np.all(max_idx[1:] != max_idx[0]) or max_idx[0] == 0:
            # If max_idx didn't change or max_idx has more than 1 unique number, keep looking for drifts
            # if np.array_equal(max_idx, max_idx_prev) or len(np.unique(max_idx)) != 1:

            # Should not be reached
            if index - no_drifts >= 500000:
                print_(f'no drifts detected from index {no_drifts} to {index}')
                return [(0, (0, len(features)))]

            index += test_batch_size
            # If index reached the end of an orbit, give this orbit a previous drift label
            if index >= orbits_idx[cur_orbit][1]:
                index = orbits_idx[cur_orbit][1]
                drift_found = True
            else:  # else keep looking for drifts
                drift_found = False
        else:
            drift_found = True

        if not drift_found:
            continue

        # End of orbit scenario
        if index == orbits_idx[cur_orbit][1]:

            print_(
                f'no drifts detected drift during orbit {orbit_numbers[cur_orbit]} - {orbits_idx[cur_orbit]} - {dates[index]}')
            print_(f'labelling orbit with previous drift label')
            if len(drift_labels) > 0:
                next_label = drift_labels[-1]
            else:
                next_label = [1]  # initial drift label

        # Found drift scenario
        else:

            print_(
                f'max_idx {max_idx_prev} -> {max_idx} [{index}] (orbit {orbit_numbers[cur_orbit]} - {orbits_idx[cur_orbit]} - {dates[index]})')
            # print_(f'prob = {prob.cpu().detach().numpy()}')
            # print_(f'discriminator output:\n{result.cpu().detach().numpy()}')
            max_idx_prev = max_idx
            next_label = generator_label

            # Drift in the middle
            if no_drifts != index:
                print_(f'no drifts detected from index {no_drifts} to {index}')
                print_(
                    f'detected drift in the middle of orbit {orbit_numbers[cur_orbit]} - {orbits_idx[cur_orbit]} - {dates[index]}')

                # If index didn't reach the crossings (approximately), give orbit a new drift label
                if (index - orbits_idx[cur_orbit][0]) / (orbits_idx[cur_orbit][1] - orbits_idx[cur_orbit][0]) < 0.5:
                    print_(f'index is below the threshold, give orbit a new label')
                    if temp_label[0] != 0:
                        # add the index of the previous drift if it was a recurring drift
                        next_label = temp_label[0]
                        print_(f'recurring drift {next_label}')
                    else:
                        print_(f'new drift {next_label}')

                else:  # else give it a previous drift label
                    print_(
                        f'index is above the threshold, give orbit a previous label')
                    if drift_labels:
                        next_label = drift_labels[-1]
                    else:
                        next_label = 0

            # Drift at the start
            else:
                if temp_label[0] != 0:
                    # add the index of the previous drift if it was a recurring drift
                    next_label = temp_label[0]
                    print_(f'recurring drift {next_label}')
                else:
                    print_(f'new drift {next_label}')

                print_(
                    f'detected drift at the start of orbit {orbit_numbers[cur_orbit]} - {orbits_idx[cur_orbit]} - {dates[index]}')

        # print_('========== START ==========')

        max_idx = max_idx[0]
        # Drift detected
        drift_indices.append(
            (orbits_idx[cur_orbit][0], orbits_idx[cur_orbit][1]))
        drift_labels.append(next_label)
        print_(f'add drift {drift_labels[-1]} {drift_indices[-1]}')

        if len(drift_labels) > 1:
            print_(f'drift from {drift_labels[-2]} to {drift_labels[-1]}')

        if max_idx != generator_label:
            # Increase the max_idx by 1 if it is above the previous drift
            if temp_label[0] <= max_idx and temp_label[0] != 0:
                max_idx += 1
            temp_label = [max_idx]
            # We reset the top layer predictions because the drift order has changed and the network should be retrained
            print_(f'discriminator.reset_top_layer()')
            discriminator.reset_top_layer()
            discriminator = discriminator.to(device)

        else:
            # If this is a new drift, label for the previous drift training dataset is the previous highest label
            # which is the generator label
            temp_label = [0]
            print_(f'discriminator.update()')
            discriminator.update()
            discriminator = discriminator.to(device)
            generator_label += 1

        generator = Generator(
            inp=features.shape[1], out=features.shape[1], sequence_length=sequence_length)
        generator = generator.to(device=device)

        generator.train()
        discriminator.train()

        print_(
            f'training dataset indices = {(drift_indices[0][0], drift_indices[-1][-1])}')
        print_(
            f'training dataset labels len = {len(drift_labels)}, unique = {np.unique(drift_labels)}')
        training_dataset = create_training_dataset(dataset=features,
                                                   indices=drift_indices,
                                                   drift_labels=drift_labels+temp_label)

        generator, discriminator = train_gan(features=training_dataset, device=device,
                                             discriminator=discriminator,
                                             generator=generator, epochs=epochs,
                                             steps_generator=steps_generator, seed=seed,
                                             batch_size=current_batch_size, max_label=generator_label,
                                             lr=lr/10, momentum=momentum, equalize=equalize,
                                             weight_decay=weight_decay, sequence_length=sequence_length)

        # Set the generator and discriminator to evaluation mode
        generator.eval()
        discriminator.eval()

        drifts_detected.append(index)

        index = orbits_idx[cur_orbit][1]
        if cur_orbit < len(orbit_numbers) - 1:
            print_(
                f'orbit change {orbit_numbers[cur_orbit]} -> {orbit_numbers[cur_orbit+1]}')
        print_(
            f'continuing drift detection from {index} (end of orbit {orbit_numbers[cur_orbit]} - {orbits_idx[cur_orbit]})')
        cur_orbit += 1

        no_drifts = index

        print_('===========================')

    # print_(
    #     f'stopping drift detection, {index} + {training_window_size} >= {len(features)}')
    drift_labels = [1] + drift_labels
    print_(
        f'stopping drift detection, {index} >= {orbits_idx[-1][-1]}')
    print_(f'len(drifts_detected) = {len(drifts_detected)}')
    print_(f'len(drift_labels) = {len(drift_labels)}')
    print_(f'len(drift_indices) = {len(drift_indices)}')
    drifts_detected.append(len(features))
    drifts = list(zip(drift_labels, drift_indices))

    for d in drifts:
        print_(f'{d[0]}: {d[1]} - orbit {orbit_numbers[orbits_idx.index(d[1])]}')

    print_(generator)
    print_(discriminator)

    return drifts


def train_clfs(features, labels, drifts):

    clfs = {}
    classes = np.unique(labels)
    weights = compute_class_weight(
        'balanced', classes=classes, y=labels)
    print_(f'weights = {weights}')

    for d in drifts:

        drift_num = d[0]
        drift_idx = d[1]

        if drift_idx[0] < len(features):

            bound = drift_idx[1]

            if bound > len(features):
                bound = len(features)
                print_(
                    f'index {drift_idx[1]} is outside of training orbits, set to {len(features)}')

            x = np.array(features[drift_idx[0]:bound, :], copy=True)
            x = x.reshape(-1, x.shape[1], 1)
            y = np.asarray(labels[drift_idx[0]:bound])

            if not drift_num in clfs:
                clfs[drift_num] = cnn(x.shape[1:])
                print_(f'create new classifier for drift {drift_num}')

            print_(
                f'training classifier for drift {drift_num} - {(drift_idx[0], bound)}...')
            clfs[drift_num].fit(x=x, y=y,
                                batch_size=16,
                                epochs=20,
                                class_weight={k: v for k,
                                              v in enumerate(weights)},
                                verbose=0)

        else:
            print_(f'{drift_idx} is outside of training orbits, ignoring')

    print_(f'trained classifiers for drifts - {list(clfs.keys())}')

    return clfs


def test_clfs(features, drifts, clfs):

    labels = list(range(len(features)))

    for d in drifts:

        drift_num = d[0]
        drift_idx = d[1]

        if drift_num in clfs:
            drift = drift_num
        else:
            drift = min(clfs.keys(), key=lambda x: abs(x-drift_num))
            print_(
                f'no classifier for drift {drift_num}, switching to {drift}')

        x = np.array(features[drift_idx[0]:drift_idx[1]], copy=True)
        x = x.reshape(-1, x.shape[1], 1)

        print_(f'testing classifier for drift {drift} - {drift_idx}...')
        pred = clfs[drift].predict(x)  # window vs step
        labels[drift_idx[0]:drift_idx[1]] = pred.argmax(axis=-1)

    return labels


def load_data(path, prev_len=0):

    split = 'training'
    if 'test' in path:
        split = 'testing'

    files = glob.glob(path)
    random.shuffle(files)

    li = []
    orbits = {}
    df_len = prev_len

    print_(f'loading {len(files)} {split} orbits...')

    for filename in files:
        df = pd.read_csv(filename, index_col=None, header=0).dropna()
        li.append(df)
        # n = int(filename.split('_')[-1].split('.')[0])
        n = df.iloc[0]['ORBIT']
        orbits[n] = (df_len, df_len + len(df.index))
        df_len = df_len + len(df.index)
        print_(
            f'loaded {split} orbit {n} - {orbits[n]} - {(df.iloc[0]["DATE"], df.iloc[-1]["DATE"])}')

    df = pd.concat(li, axis=0, ignore_index=True)

    return df, orbits


def calc_stats(cm):

    precision = []
    recall = []
    f1 = []

    for i, _ in enumerate(cm):
        tp = cm[i][i]
        fp = 0
        fn = 0

        for j, _ in enumerate(cm[i]):
            if j != i:
                fp += cm[j][i]
                fn += cm[i][j]

        precision.append(tp / (tp + fp) if tp + fp != 0 else 0)
        recall.append(tp / (tp + fn) if tp + fn != 0 else 0)
        f1.append(2 * precision[i] * recall[i] / (precision[i] +
                  recall[i]) if precision[i] + recall[i] != 0 else 0)

    print_(f'precision = {precision}')
    print_(f'recall = {recall}')
    print_(f'F1 score = {f1}')


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


def plot_orbit(df, orbits, title, draw=[1, 3]):

    global plot_format

    df['B_tot'] = (df['BX_MSO']**2 + df['BY_MSO']**2 + df['BZ_MSO']**2)**0.5
    colors = {0: 'red', 1: 'green', 2: 'yellow', 3: 'blue', 4: 'purple'}

    label_col = 'LABEL'
    if 'pred' in title:
        label_col = 'LABEL_PRED'

    for n in orbits:

        df_orbit = df.iloc[orbits[n][0]:orbits[n][1]]
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

        fig.update_layout({'title': f'{title}_orbit{n}'})

        if 'png' in plot_format:
            fig.write_image(
                f'../logs/{folder}/{title}/fig_{n}.png')
        if 'html' in plot_format:
            fig.write_html(
                f'../logs/{folder}/{title}/fig_{n}.html')


# %% setup

fptr = None
dataset = 'messenger'
folder = f'{str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))}_{sys.argv[1:]}'

plots = ''
if len(sys.argv) > 2:
    plots = sys.argv[2]

plot_format = 'png'
if len(sys.argv) > 3:
    plot_format = sys.argv[3]

skip = False
if len(sys.argv) > 4:
    skip = bool(int(sys.argv[4]))

# Select dataset split
set_number = 1
if len(sys.argv) > 1:
    set_number = int(sys.argv[1])
print_(f'set_number: {set_number}')

# Set the number of epochs the GAN should be trained
epochs = 20  # 50
print_(f'epochs: {epochs}')

# 1/factor will be the amount of instances of previous drifts taken for training
repeat_factor = 5  # 10 test this
print_(f'repeat_factor: {repeat_factor}')

# Equalize the number of training instances across different drifts
equalize = True

# How far in to the past is required for generating current data
sequence_length = 10
print_(f'sequence_length: {sequence_length}')
# For the collate function to split the rows accordingly
seq_len = sequence_length

# Steps for generator training
steps_generator = 20
print_(f'steps_generator: {steps_generator}')

# Set the batch_size for DataLoader
batch_size = 8
print_(f'batch_size: {batch_size}')
generator_batch_size = 2
print_(f'generator_batch_size: {generator_batch_size}')
# Number of instances that should have the same label for a drift to be confirmed
test_batch_size = 4
print_(f'test_batch_size: {test_batch_size}')

# Set the learning rate
lr = 0.025  # Changed to Adadelta with a default learning rate of 1
print_(f'learning rate: {lr}')

# Set the weight decay rate
weight_decay = 0.000000
print_(f'weight_decay: {weight_decay}')

# Set a random seed for the experiment
seed = np.random.randint(65536)

np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)

# Get the device the experiment will run on
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

print_(
    f'the seed for the current execution is {seed} for dataset {dataset} with device {device}')


# %% load data

df_train, orbits_train = load_data(f'../data/orbits{set_number}/train/*.csv')
df_test, orbits_test = load_data(
    f'../data/orbits{set_number}/test/*.csv', prev_len=len(df_train.index))


# %% select data

feats = ['X_MSO', 'Y_MSO', 'Z_MSO', 'BX_MSO', 'BY_MSO', 'BZ_MSO', 'DBX_MSO', 'DBY_MSO', 'DBZ_MSO', 'RHO_DIPOLE', 'PHI_DIPOLE', 'THETA_DIPOLE',
         'BABS_DIPOLE', 'BX_DIPOLE', 'BY_DIPOLE', 'BZ_DIPOLE', 'RHO', 'RXY', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'VABS', 'D', 'COSALPHA', 'EXTREMA']

with open('../data/features.txt', 'r') as f:
    feats = [line.strip() for line in f]
print_(f'selected features: {feats}')

df_train = select_features(df_train, feats)
df_test = select_features(df_test, feats)
df_all = pd.concat([df_train, df_test])
orbits_all = {**orbits_train, **orbits_test}

dates = df_all.iloc[:, 0].values.tolist()
labels_train_true = df_train.iloc[:, -1].values.tolist()
labels_test_true = df_test.iloc[:, -1].values.tolist()
labels_all_true = df_all.iloc[:, -1].values.tolist()

# standardization
features_all = df_all.iloc[:, 1:-1].values
mean = np.mean(features_all, axis=1).reshape(features_all.shape[0], 1)
std = np.std(features_all, axis=1).reshape(features_all.shape[0], 1)
features_all = (features_all - mean) / (std + 0.000001)
features_train = features_all[0:len(labels_train_true)]
features_test = features_all[-len(labels_test_true):]

print_(f'total size = {len(features_all)}')
print_(f'training set size = {len(features_train)}')
print_(f'testing set size = {len(features_test)}')
print_(f'training indices = [0:{len(features_train)}]')
print_(f'testing indices = [{len(features_train)}:{len(features_all)}]')


# %% training GAN

"""
# Min max scaling
min_features = np.min(features, axis=1)
features = features - np.reshape(min_features, newshape=(min_features.shape[0], 1))
max_features = np.max(features, axis=1)
max_features = np.reshape(max_features, newshape=(max_features.shape[0], 1)) + 0.000001
features = features / max_features
"""

if skip:
    drifts = [(0, (0, len(features_all)))]
else:
    t1 = time.perf_counter()
    drifts = detect_drifts(features=features_all, orbits=orbits_all,
                           dates=dates, device=device, epochs=epochs,
                           steps_generator=steps_generator, seed=seed,
                           batch_size=batch_size, lr=lr, momentum=0.9,
                           weight_decay=weight_decay, test_batch_size=test_batch_size,
                           generator_batch_size=generator_batch_size, equalize=equalize,
                           sequence_length=sequence_length, repeat_factor=repeat_factor)
    t2 = time.perf_counter()
    print_(f'drift detection time is {t2 - t1:.2f} seconds')


# %% training classifiers

t1 = time.perf_counter()
clfs = train_clfs(features=features_train,
                  labels=labels_train_true, drifts=drifts)
t2 = time.perf_counter()
print_(f'training time is {t2 - t1:.2f} seconds')


# %% testing classifiers

t1 = time.perf_counter()
all_pred = test_clfs(features_all, drifts, clfs)
t2 = time.perf_counter()
print_(f'testing time is {t2 - t1:.2f} seconds')

df_all['LABEL_PRED'] = all_pred
labels_train_pred = all_pred[:len(features_train)]
labels_test_pred = all_pred[-len(features_test):]

# Fit to already predicted features
# test_pred, clf = predict_and_partial_fit(
#     clf, features=features_test, labels=test_true, classes=np.unique(test_true))


# %% pad missing labels

if len(labels_train_true) < len(df_train.index):
    print_(
        f'padding training set true values with [{labels_train_true[-1]}] * {len(df_train.index) - len(labels_train_true)}')
    labels_train_true += [labels_train_true[-1]] * \
        (len(df_train.index) - len(labels_train_true))

if len(labels_train_pred) < len(df_train.index):
    print_(
        f'padding training set predictions with [{labels_train_pred[-1]}] * {len(df_train.index) - len(labels_train_pred)}')
    labels_train_pred += [labels_train_pred[-1]] * \
        (len(df_train.index) - len(labels_train_pred))


# %% evaluation

for n in orbits_all:
    split = 'train'
    if n in orbits_test:
        split = 'test'
    f1 = precision_recall_fscore_support(labels_all_true[orbits_all[n][0]:orbits_all[n][1]],
                                         all_pred[orbits_all[n][0]:orbits_all[n][1]],
                                         average=None,
                                         labels=np.unique(labels_all_true[orbits_all[n][0]:orbits_all[n][1]]))[2]
    print_(f'{split} orbit {n} {orbits_all[n]} f-score - {f1}')

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


# %% plots

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


# %% close log file

if fptr is not None:
    fptr.close()
    fptr = None


# %%
