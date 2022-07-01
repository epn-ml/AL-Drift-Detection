# %% Imports

import glob
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import wandb
from torch import nn
from torch.autograd import Variable
from torch.nn import Linear, Module, ReLU, Sequential
from torch.optim import Adadelta
from torch.utils.data import DataLoader

from util import load_data, print_f, select_features

global seq_len
global fptr


# %% Functions

# Wrapper for print function
def print_(print_str, with_date=True):

    global fptr
    print_f(fptr, print_str, with_date)
    if with_date:
        print(f'{str(datetime.now())}: {print_str}', flush=True)
    else:
        print(print_str, flush=True)


# Generator class
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


# Discriminator class
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


# Train discriminator on real and generated data
def train_discriminator(real_data, fake_data, discriminator, generator, optimizer, loss_fn,
                        generator_labels, device):

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
        wandb.log({"generated_data_loss": generated_data_loss})

        # Add the loss and compute back prop
        total_iter_loss = generated_data_loss + real_data_loss
        total_iter_loss.backward()
        wandb.log({"total_discriminator_loss": total_iter_loss})

        # Update parameters
        optimizer.step()

    return discriminator


# Train generator on output of discriminator
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
        wandb.log({"total_generator_loss": total_generator_loss})

        # Back prop and parameter update
        total_generator_loss.backward()
        optimizer.step()
        epoch_loss += total_generator_loss.item()

    return generator


# Concatenate feature vectors into one vector for entire dataset
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
def create_training_dataset(dataset, indices, drift_labels, max_length=100):

    removed = {}
    while len(drift_labels) > max_length:
        indices = indices.copy()
        drift_labels = drift_labels.copy()
        u, c = np.unique(drift_labels, return_counts=True)
        i = drift_labels.index(u[np.argmax(c)])

        if drift_labels[i] in removed:
            removed[drift_labels[i]] += 1
        else:
            removed[drift_labels[i]] = 1

        del indices[i]
        del drift_labels[i]

    # if len(removed) > 0:
    #     print_(f'removed labels = {removed}')

    # print_(
    #     f'training dataset indices = {(indices[0][0], indices[-1][-1])}')
    # print_(
    #     f'training dataset labels len = {len(drift_labels)}, unique =\n{np.array(np.unique(drift_labels, return_counts=True))}')

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

    training_dataset = np.hstack((dataset[indices[0][0]:indices[0][1]],
                                  np.ones((indices[0][1]-indices[0][0], 1)) * modified_drift_labels[0]))
    for idx in range(1, len(modified_drift_labels)):
        training_dataset = np.vstack((training_dataset, np.hstack((dataset[indices[idx][0]:indices[idx][1]],
                                      np.ones((indices[idx][1]-indices[idx][0], 1)) * modified_drift_labels[idx]))))

    return training_dataset


# Equalize number of drift labels used for training
def equalize_classes(features, max_count=500):

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


# Concatenate feature vectors into one vector
def concat_feature(data, idx, sequence_len=2):

    if idx < len(data) - sequence_len - 1:
        return data[idx:idx + sequence_len + 1, :].flatten()
    if idx == len(data) - sequence_len - 1:
        return data[idx:, :].flatten()
    return np.hstack((data[idx:idx + sequence_len, :].flatten(), data[sequence_length - 1]))


# Equalize before concatenating
def equalize_and_concatenate(features, max_count=500, sequence_len=2):

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


# Train GAN (discriminator and generator)
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

    if equalize:
        # Equalize and concatenate at the same time
        concatenated_data = equalize_and_concatenate(
            features, sequence_len=sequence_length)
        features = equalize_classes(features)
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

    # print_(f'training GAN...')
    for epochs_trained in range(epochs):
        discriminator = train_discriminator(real_data=real_data, fake_data=generator_data, discriminator=discriminator,
                                            generator=generator, optimizer=optimizer_discriminator,
                                            loss_fn=loss_discriminator, generator_labels=generator_label, device=device)

        generator = train_generator(data_loader=generator_data, discriminator=discriminator, generator=generator,
                                    optimizer=optimizer_generator, loss_fn=loss_generator, loss_mse=loss_mse_generator,
                                    steps=steps_generator, device=device)

    return generator, discriminator


# Train GAN and detect drifts
def detect_drifts(df, device, epochs=100, steps_generator=100, equalize=True, test_batch_size=4,
                  seed=0, batch_size=8, lr=0.001, momentum=0.9, weight_decay=0.0005,
                  generator_batch_size=1, sequence_length=2):

    # Standardization
    # features = df.iloc[:, 1:-2].values
    # print_(f'features:\n{features[:5]}')
    # print_(f'mean:\n{features.mean(axis=0)}')

    # mean = np.mean(features, axis=1).reshape(features.shape[0], 1)
    # std = np.std(features, axis=1).reshape(features.shape[0], 1)
    # features = (features - mean) / (std + 0.000001)
    # print_(f'standardized:\n{features[:5]}')
    # print_(f'mean:\n{features.mean(axis=0)}')
    # print_(f'total size = {len(features)}')

    # df = df.loc[df['LABEL'] == 0]
    df_features = df.iloc[:, 1:-2]
    print_(f'features:\n{df.columns}')

    df.iloc[:, 1:-2] = (df_features - df_features.mean()) / df_features.std()
    df_features = df.iloc[:, 1:-2]

    for col in df_features:
        df_features[col] = df_features[col].rolling(5000, min_periods=1).mean()
    df.iloc[:, 1:-2] = df_features
    print_(f'rolling mean:\n{df_features.head()}')
    print_(f'total size = {len(df.index)}')
    features = df.iloc[:, 1:-2].values

    orbit_numbers = pd.unique(df['ORBIT']).tolist()
    print_(f'total number of orbits = {len(orbit_numbers)}')
    orbits_idx = []
    for orbit in orbit_numbers:
        idx = df.loc[df['ORBIT'] == orbit].index
        orbits_idx.append((idx[0], idx[-1] + 1))
        print_(
            f'{orbit} - {orbits_idx[-1]} - ({df["DATE"].iloc[idx[0]]}, {df["DATE"].iloc[idx[-1]]})', with_date=False)
    for i in range(1, len(orbits_idx)-1):
        if orbits_idx[i][0] != orbits_idx[i-1][1]:
            orbits_idx[i] = (orbits_idx[i-1][1], orbits_idx[i][1])
            print_(f'replaced bad orbit 1st idx: {orbits_idx[i]}')
        if orbits_idx[i][1] != orbits_idx[i+1][0]:
            orbits_idx[i] = (orbits_idx[i][0], orbits_idx[i+1][0])
            print_(f'replaced bad orbit 2nd idx: {orbits_idx[i]}')

    # Initial orbit with known drift
    drift_indices = [orbits_idx[0]]
    cur_orbit = 1
    drift_labels = []
    drift_orbits = {orbit_numbers[0]: 1}
    # queue_idx = orbits_idx[1:100].copy()
    # queue_labels = 12*[1] + 14*[2] + 15*[3] + \
    #     8*[4] + 13*[5] + 14*[6] + 12*[7] + 11*[8]

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

    wandb.watch(discriminator)

    generator.move(device=device)

    # Set the models to the device
    generator = generator.to(device=device)
    discriminator = discriminator.to(device=device)

    temp_label = [0]
    initial_epochs = epochs * 2

    # Create training dataset
    # print_(f'training dataset indices = {drift_indices}')
    # print_(f'training dataset labels  = {drift_labels+temp_label}')
    training_dataset = create_training_dataset(
        dataset=features, indices=drift_indices, drift_labels=drift_labels+temp_label)

    generator, discriminator = train_gan(features=training_dataset, device=device, discriminator=discriminator,
                                         generator=generator, epochs=initial_epochs, steps_generator=steps_generator,
                                         seed=seed, batch_size=batch_size, lr=lr, momentum=momentum, equalize=equalize,
                                         max_label=generator_label, generator_batch_size=generator_batch_size,
                                         weight_decay=weight_decay, sequence_length=sequence_length)

    generator.eval()
    discriminator.eval()

    index = orbits_idx[cur_orbit][0]

    no_drifts = index
    max_idx_prev = np.array(test_batch_size * [0])
    end_orbit = 13
    prev_drift = -1

    print_(
        f'starting drift detection from index = {index} (orbits {orbit_numbers[cur_orbit]} - {orbit_numbers[end_orbit-1]})')
    print_('===========================')

    while index < orbits_idx[-1][-1]:

        # Pre-train
        if cur_orbit < 100:
            max_idx = max_idx_prev + 1
            prob = np.full(max_idx.shape, 1)
        else:
            data = features[index:index + test_batch_size]
            result = discriminator(torch.Tensor(
                data).to(torch.float).to(device))
            prob, max_idx = torch.max(result, dim=1)
            prob = prob.cpu().detach().numpy()
            max_idx = max_idx.cpu().detach().numpy()

        if not np.array_equal(max_idx, max_idx_prev):
            # print_(
            #     f'max_idx {max_idx_prev} -> {max_idx} [{index}]', with_date=False)
            # print_(f'prob = {prob.cpu().detach().numpy()}')
            # print_(f'discriminator output:\n{result.cpu().detach().numpy()}')
            max_idx_prev = max_idx

        drift_found = False

        if np.all(max_idx[1:] != max_idx[0]) or max_idx[0] == 0:

            # Should not be reached
            if index - no_drifts >= 500000:
                print_(
                    f'no drifts detected from index {no_drifts} to {index}, terminating')
                return dict(zip(orbit_numbers, [1]*len(orbit_numbers)))

            index += test_batch_size
            # If index reached the end of an orbit, give it a previous drift label
            if index >= orbits_idx[cur_orbit][1]:
                index = orbits_idx[cur_orbit][1]
                drift_found = True
            else:  # else keep looking for drifts
                drift_found = False
        else:
            drift_found = True

        if not drift_found:
            continue

        # Drift detected
        # End of orbit scenario
        if index >= orbits_idx[cur_orbit][1]:

            next_drift = drift_orbits[orbit_numbers[cur_orbit-1]]
            end_orbit = cur_orbit + 1
            if len(drift_labels) > 0:
                next_label = drift_labels[-1]
            else:
                next_label = 1  # initial drift label
            print_(
                f'no drifts detected during orbit {orbit_numbers[cur_orbit]}')

        else:

            next_drift = max_idx[0]
            # print_(f'orbit {cur_orbit} / {len(orbit_numbers)}')
            # print_(
            #     f'{index} / {orbits_idx[-1][-1]} {100 * index / orbits_idx[-1][-1]:.2f}%')
            next_label = generator_label

            if temp_label[0] != 0:
                # add the index of the previous drift if it was a recurring drift
                next_label = temp_label[0]
                # print_(f'recurring drift {next_label} (temp_label[0])')
            # else:
            #     print_(f'new drift {next_label} (generator_label)')

            # Drift in the middle
            # if no_drifts != index:
            #     print_(
            #         f'no drifts detected from index {no_drifts} to {index}')

            # Drift at the start
            # else:
            #     print_(
            #         f'detected drift at the start of orbit {orbit_numbers[cur_orbit]} - {orbits_idx[cur_orbit]}')

        max_idx = max_idx[0]
        prob = prob[0]
        if max_idx != generator_label:
            # Increase the max_idx by 1 if it is above the previous drift
            if temp_label[0] <= max_idx and temp_label[0] != 0:
                # print_(
                #     f'temp_label[0] {temp_label[0]} <= max_idx {max_idx}, max_idx += 1')
                if next_drift == max_idx and end_orbit != cur_orbit + 1:
                    next_drift += 1
                max_idx += 1
            # We reset the top layer predictions because the drift order has changed and the network should be retrained
            # print_(
            #     f'discriminator.reset_top_layer(), temp_label[0] {temp_label[0]} -> max_idx {max_idx}')
            temp_label = [max_idx]
            discriminator.reset_top_layer()
            discriminator = discriminator.to(device)
            # print_(
            #     f'Previous drift {max_idx} occured at {index} (orbits {orbit_numbers[cur_orbit]} - {orbit_numbers[end_orbit-1]})')

        else:
            # If this is a new drift, label for the previous drift training dataset is the previous highest label
            # which is the generator label
            # print_(
            #     f'discriminator.update(), temp_label[0] {temp_label[0]} -> 0, generator_label {generator_label} += 1')
            temp_label = [0]
            discriminator.update()
            discriminator = discriminator.to(device)
            generator_label += 1

        if next_drift == prev_drift and end_orbit != cur_orbit + 1:
            next_drift = next_label

        new_orbits = orbit_numbers[cur_orbit:end_orbit]
        new_drift_orbits = dict(zip(new_orbits, [next_drift]*len(new_orbits)))
        drift_orbits = {**drift_orbits, **new_drift_orbits}
        prev_drift = next_drift
        print_(
            f'{end_orbit}/{len(orbit_numbers)} orbits {new_orbits[0]} - {new_orbits[-1]} ({end_orbit-cur_orbit}) -- drift {next_drift}, prob {prob}')
        wandb.log({"orbit": end_orbit})
        wandb.log({"drift": next_drift})

        drift_indices.append(
            (orbits_idx[cur_orbit][0], orbits_idx[end_orbit-1][1]))
        drift_labels.append(next_label)

        generator = Generator(
            inp=features.shape[1], out=features.shape[1], sequence_length=sequence_length)
        generator = generator.to(device=device)

        generator.train()
        discriminator.train()

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

        index = orbits_idx[end_orbit-1][1]
        no_drifts = index
        # if cur_orbit + orbits_max < len(orbit_numbers):
        #     print_(
        #         f'orbit change {orbit_numbers[cur_orbit]} -> {orbit_numbers[cur_orbit+orbits_max]}')

        cur_orbit = end_orbit
        end_orbit = cur_orbit + 1
        if cur_orbit < 100:
            orbits_max = 21
        else:
            orbits_max = random.randrange(14, 22)

        while end_orbit < len(orbit_numbers):
            # 6 - max difference between orbits in the same drift
            if abs(orbit_numbers[end_orbit] - orbit_numbers[end_orbit-1]) > 6:
                break
            if orbit_numbers[end_orbit] - orbit_numbers[cur_orbit] >= orbits_max:
                break
            end_orbit += 1

        # print_('===========================')

    drift_labels = [1] + drift_labels
    print_(
        f'stopping drift detection, {index} >= {orbits_idx[-1][-1]}')
    print_(f'len(drifts_detected) = {len(drifts_detected)}')
    print_(f'len(drift_labels) = {len(drift_labels)}')
    print_(f'len(drift_indices) = {len(drift_indices)}')
    drifts_detected.append(len(features))
    drifts = list(zip(drift_labels, drift_indices))

    for d in drifts:
        print_(f'indices {d[1]} -- drift {d[0]}', with_date=False)

    print_(generator)
    print_(discriminator)

    return drift_orbits


# %% Setup

logs = sys.argv[1]
dataset = int(sys.argv[2])
if not os.path.exists(logs):
    os.makedirs(logs)

fptr = open(f'{logs}/log_set{dataset}.txt', 'w')
print_(f'dataset: {dataset}')

# Set the number of epochs the GAN should be trained
epochs = 20  # 50
print_(f'epochs: {epochs}')

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

device_name = 'cuda'
if len(sys.argv) > 3:
    device_name = sys.argv[3]

# Get the device the experiment will run on
device = torch.device(device_name if torch.cuda.is_available() else 'cpu')

print_(
    f'the seed for the current execution is {seed} for MESSENGER dataset with device {device}')

wandb.init(project="gan", entity="irodionr", config={
    "sequence_length": 10,
    "steps_generator": 20,
    "batch_size": 8,
    "generator_batch_size": 2,
    "test_batch_size": 4,
    "epochs": 20,
    "weight_decay": 0.000000,
    "learning_rate": 0.025
})

# %% Load data

files = glob.glob('data/orbits/*.csv')
files.sort(key=lambda x: int(''.join(i for i in x if i.isdigit())))

if dataset == 1:
    pass  # full dataset
elif dataset == 2:
    files = files[460:760]
elif dataset == 3:
    idx = random.randrange(0, len(files) // 2 - 300)
    files = files[idx:idx+300]
elif dataset == 4:
    idx = random.randrange(len(files) // 2, len(files) - 400)
    files = files[idx:idx+400]
elif dataset == 5:
    idx = random.randrange(0, len(files) - 1000)
    files = files[idx:idx+1000]


# %% Select data

df = load_data(files, add_known_drifts=True)
df = select_features(df, 'data/features_gan.txt')
print_(f'selected data:\n{df.columns}')

# %% Training GAN

t1 = time.perf_counter()
drift_orbits = detect_drifts(df=df, device=device, epochs=epochs, steps_generator=steps_generator,
                             seed=seed, batch_size=batch_size, lr=lr, momentum=0.9,
                             weight_decay=weight_decay, test_batch_size=test_batch_size,
                             generator_batch_size=generator_batch_size, equalize=equalize,
                             sequence_length=sequence_length)
t2 = time.perf_counter()
print_(f'drift detection time is {t2 - t1:.2f} seconds')

with open(f'{logs}/drifts_set{dataset}.txt', 'w') as drifts_file_log:
    for orbit in drift_orbits:
        drifts_file_log.write(
            f'{orbit} {drift_orbits[orbit]}\n')

# %% Close log file

if fptr is not None:
    fptr.close()
    fptr = None
