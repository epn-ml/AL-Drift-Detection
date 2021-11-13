# %% imports

from torch.optim import Adadelta
from torch import nn
from torch.nn import Sequential, Linear, Dropout, ReLU, Module
from torch.utils.data import DataLoader
from torch.autograd import Variable
from skmultiflow.trees import HoeffdingTreeClassifier
from time import time
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

import glob
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go

global seq_len
global fptr
global dataset


# %% functions

# Function to write the log to disk also
def print_(print_str):
    global fptr
    global dataset
    if fptr is None:
        name = '../logs/log-' + \
            str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S')) + '.txt'
        fptr = open(name, "w")

    fptr.write(str(datetime.now()) + ": " + str(print_str) + "\n")
    print(str(datetime.now()) + ": " + str(print_str))


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
        output = self.net(x_.reshape(x_.shape[0], x_.shape[1] * x_.shape[2]))
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


def fit_and_predict(clf, features, labels, classes):
    predicted = np.empty(shape=len(labels))
    predicted[0] = clf.predict([features[0]])
    clf.reset()
    ft = [features[0]]
    lb = [labels[0]]
    clf.partial_fit(ft, lb, classes=classes)
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


def create_training_dataset(dataset, indices, drift_labels):

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

    # This data contains the current vector and next vector
    concatenated_data = concatenate_features(
        features, sequence_len=sequence_length)

    if equalize:
        features = equalize_classes(features)
        concatenated_data = equalize_classes(concatenated_data)

    # Define the data loader for training
    real_data = DataLoader(features, batch_size=batch_size,
                           shuffle=True, collate_fn=collate)
    generator_data = DataLoader(concatenated_data, batch_size=generator_batch_size, shuffle=False,
                                collate_fn=collate_generator)

    # This is the label for new drifts (any input other than the currently learned distributions)
    generator_label = ones * max_label

    for epochs_trained in range(epochs):
        discriminator = train_discriminator(real_data=real_data, fake_data=generator_data, discriminator=discriminator,
                                            generator=generator, optimizer=optimizer_discriminator,
                                            loss_fn=loss_discriminator, generator_labels=generator_label, device=device)

        generator = train_generator(data_loader=generator_data, discriminator=discriminator, generator=generator,
                                    optimizer=optimizer_generator, loss_fn=loss_generator, loss_mse=loss_mse_generator,
                                    steps=steps_generator, device=device)
    return generator, discriminator


def process_data(features, labels, features_window, labels_window, device, epochs=100, steps_generator=100, equalize=True, test_batch_size=4,
                 seed=0, batch_size=8, lr=0.001, momentum=0.9, weight_decay=0.0005, training_window_size=100,
                 generator_batch_size=1, sequence_length=2, repeat_factor=4):

    # Set the seed
    import random

    random.seed(seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.set_deterministic(True)

    current_batch_size = batch_size

    y_pred = []
    y_true = []
    clf = HoeffdingTreeClassifier()

    classes = np.unique(labels)
    x = features[:training_window_size, :]
    y = labels[:training_window_size]

    drifts_detected = []
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

    drift_indices = [(0, training_window_size)]  # Initial training window
    drift_labels = []

    temp_label = [0]

    initial_epochs = epochs * 2

    predicted, clf = fit_and_predict(
        clf=clf, features=x, labels=y, classes=classes)
    y_pred = y_pred + predicted.tolist()
    y_true = y_true + y

    # Create training dataset
    training_dataset = create_training_dataset(
        dataset=features, indices=drift_indices, drift_labels=[0])

    generator, discriminator = train_gan(features=training_dataset, device=device, discriminator=discriminator,
                                         generator=generator, epochs=initial_epochs, steps_generator=steps_generator,
                                         seed=seed, batch_size=batch_size, lr=lr, momentum=momentum, equalize=equalize,
                                         max_label=generator_label, generator_batch_size=generator_batch_size,
                                         weight_decay=weight_decay, sequence_length=sequence_length)

    index = training_window_size

    generator.eval()
    discriminator.eval()

    while index + training_window_size < len(features):

        data = features[index:index + test_batch_size]
        data_labels = labels[index:index + test_batch_size]
        result = discriminator(torch.Tensor(data).to(torch.float).to(device))
        prob, max_idx = torch.max(result, dim=1)
        max_idx = max_idx.cpu().detach().numpy()
        if np.all(max_idx != max_idx[0]) or max_idx[0] == 0:
            predicted, clf = predict_and_partial_fit(clf=clf, features=data, labels=data_labels,
                                                     classes=classes)
            y_pred = y_pred + predicted.tolist()
            y_true = y_true + data_labels
            index += test_batch_size
            continue

        max_idx = max_idx[0]
        # Drift detected
        drift_indices.append((index, index+training_window_size))

        if temp_label[0] != 0:
            # add the index of the previous drift if it was a recurring drift
            drift_labels.append(temp_label[0])

        else:
            drift_labels.append(generator_label)

        if max_idx != generator_label:
            # Increase the max_idx by 1 if it is above the previous drift
            if temp_label[0] <= max_idx and temp_label[0] != 0:
                max_idx += 1
            temp_label = [max_idx]
            # We reset the top layer predictions because the drift order has changed and the network should be retrained
            discriminator.reset_top_layer()
            discriminator = discriminator.to(device)
            print_('Previous drift %d occurred at index %d.' %
                   (max_idx, index))

        else:
            # If this is a new drift, label for the previous drift training dataset is the previous highest label
            # which is the generator label
            temp_label = [0]
            discriminator.update()
            discriminator = discriminator.to(device)
            generator_label += 1

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

        # Set the indices for the training window
        training_idx_start = index
        training_idx_end = training_idx_start + training_window_size

        # If a previous drift has occurred use those for training the classifier but not predict on them
        if temp_label[0] != 0:
            clf.reset()
            for indices, label in zip(drift_indices[:-1], drift_labels):
                if label == temp_label[0]:
                    rows = features[indices[0]:indices[1], :]
                    targets = labels[indices[0]:indices[1]]
                    # Randomly sample .1 of the data
                    len_indices = list(range(0, rows.shape[0]))
                    chosen_indices = random.sample(
                        len_indices, int(rows.shape[0] / repeat_factor))
                    # Append rows and targets. Do random.sample and then split the matrix
                    rows = rows[chosen_indices]
                    targets = [targets[x] for x in chosen_indices]
                    clf.partial_fit(X=rows, y=targets, classes=classes)

            predicted, clf = predict_and_partial_fit(clf=clf, features=features[training_idx_start:training_idx_end, :],
                                                     labels=labels[training_idx_start:training_idx_end],
                                                     classes=classes)

        else:
            predicted, clf = fit_and_predict(clf=clf, features=features[training_idx_start:training_idx_end, :],
                                             labels=labels[training_idx_start:training_idx_end],
                                             classes=classes)
        """
        predicted, clf = fit_and_predict(clf=clf, features=features[training_idx_start:training_idx_end, :],
                                         labels=labels[training_idx_start:training_idx_end],
                                         classes=classes)
        """
        # Add the predicted and true values to the list
        predicted = predicted.tolist()
        y_pred = y_pred + predicted
        y_true = y_true + labels[training_idx_start:training_idx_end]

        drifts_detected.append(index)

        print_('index = %d' % index)
        index += training_window_size

    print_(generator)
    print_(discriminator)

    # Test on the remaining features
    # features_window = features[index:, :]
    # labels_window = labels[index:]
    # y_hat, clf = predict_and_partial_fit(
    #     clf, features=features_window, labels=labels_window, classes=classes)
    # y_pred = y_pred + y_hat.tolist()
    # y_true = y_true + labels_window

    # save model
    torch.save(
        clf, f'../logs/model_{str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))}_clf.pth')
    torch.save(
        generator, f'../logs/model_{str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))}_gen.pth')
    torch.save(discriminator,
               f'../logs/model_{str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))}_dis.pth')
    torch.save(generator.state_dict(
    ), f'../logs/model_{str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))}_gen_state.pth')
    torch.save(discriminator.state_dict(
    ), f'../logs/model_{str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))}_dis_state.pth')

    y_hat, clf = predict_and_partial_fit(
        clf, features=features_window, labels=labels_window, classes=classes)
    # y_pred = y_pred + y_hat.tolist()
    # y_true = y_true + labels_window

    print_(f'drift_indices = {drift_indices}')
    print_(f'drift_labels = {drift_labels}')

    return y_pred, y_true, y_hat.tolist(), labels_window, drifts_detected


def load_data(path):

    files = glob.glob(path)
    li = []
    breaks = []

    for filename in files:
        df = pd.read_csv(filename, index_col=None, header=0)
        breaks.append(df.iloc[0]['DATE'])
        breaks.append(df.iloc[-1]['DATE'])
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)

    return df.dropna(), sorted(breaks)[1:-1]


def calc_stats(cm):

    precision = []
    recall = []
    f1 = []

    for i in range(len(cm)):
        tp = cm[i][i]
        fp = 0
        fn = 0

        for j in range(len(cm[i])):
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

    drop_col = ['Unnamed: 0', 'DATE', 'X_MSO', 'Y_MSO', 'Z_MSO', 'BX_MSO', 'BY_MSO', 'BZ_MSO', 'DBX_MSO', 'DBY_MSO', 'DBZ_MSO', 'RHO_DIPOLE', 'PHI_DIPOLE', 'THETA_DIPOLE',
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

    fig = plot_field(df)
    
    for i in range(0, len(breaks), 2):
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=[breaks[i], breaks[i+1]])
            ]
        )

    label_col = 'LABEL'
    if labels != None:
        df['LABEL_PRED'] = labels
        label_col = 'LABEL_PRED'

    colors = {0: 'red', 1: 'green', 2: 'yellow', 3: 'blue', 4: 'purple'}

    for i in draw:
        for _, row in df[df[label_col] == i].iterrows():
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
        f'../logs/fig_{str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))}_{title}.html')


# %% setup

fptr = None
dataset = 'messenger'
# Set the number of training instances
training_window_size = 1000
print_(f'training_window_size: {training_window_size}')
# Set the number of epochs the GAN should be trained
epochs = 20
print_(f'epochs: {epochs}')

# 1/factor will be the amount of instances of previous drifts taken for training
repeat_factor = 10
print_(f'repeat_factor: {repeat_factor}')

# Equalize the number of training instances across different drifts
equalize = True

# How far in to the past is required for generating current data
sequence_length = 10
print_(f'sequence_length: {sequence_length}')
# For the collate function to split the rows accordingly
seq_len = sequence_length

# Steps for training
steps_generator = 50
print_(f'steps_generator: {steps_generator}')

# Set the batch_size
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

# Get the device the experiment will run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print_('The seed for the current execution is %d for dataset %s with device %s' % (
    seed, dataset, device))


# %% load data

df_train, breaks_train = load_data('../data/labelled_orbits/train/*.csv')
df_test, breaks_test = load_data('../data/labelled_orbits/test/*.csv')


# %% select data

feats = ['DATE', 'BX_MSO', 'BY_MSO', 'BZ_MSO', 'DBX_MSO', 'DBY_MSO', 'DBZ_MSO', 'RHO', 'BABS_DIPOLE',
         'RHO_DIPOLE', 'BX_DIPOLE', 'BY_DIPOLE', 'BZ_DIPOLE', 'COSALPHA', 'EXTREMA']
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
print(f'features_train: {len(features_train)}')
print(f'labels_train: {len(labels_train)}')

features_test = df_test.iloc[:, 1:-1].values
labels_test = df_test.iloc[:, -1].values.tolist()
mean = np.mean(features_test, axis=1).reshape(features_test.shape[0], 1)
std = np.std(features_test, axis=1).reshape(features_test.shape[0], 1)
features_test = (features_test - mean)/(std + 0.000001)
u, c = np.unique(labels_test, return_counts=True)
print_(dict(zip(u, c)))
print(f'features_test: {len(features_test)}')
print(f'labels_test: {len(labels_test)}')


# %% training

"""
# Min max scaling
min_features = np.min(features, axis=1)
features = features - np.reshape(min_features, newshape=(min_features.shape[0], 1))
max_features = np.max(features, axis=1)
max_features = np.reshape(max_features, newshape=(max_features.shape[0], 1)) + 0.000001
features = features / max_features
"""

t1 = time()
train_pred, train_true, test_pred, test_true, drifts_detected = process_data(features=features_train, labels=labels_train, features_window=features_test, labels_window=labels_test, device=device, epochs=epochs,
                                                                             steps_generator=steps_generator, seed=seed,
                                                                             batch_size=batch_size, lr=lr, momentum=0.9,
                                                                             weight_decay=weight_decay, test_batch_size=test_batch_size,
                                                                             training_window_size=training_window_size,
                                                                             generator_batch_size=generator_batch_size, equalize=equalize,
                                                                             sequence_length=sequence_length, repeat_factor=repeat_factor)
t2 = time()


# %% pad missing labels

if len(train_pred) < len(df_train.index):
    train_pred += [train_pred[-1]] * (len(df_train.index) - len(train_pred))
if len(train_true) < len(df_train.index):
    train_true += [train_true[-1]] * (len(df_train.index) - len(train_true))

print(f'train_pred: {len(train_pred)}')
print(f'train_true: {len(train_true)}')
print(f'test_pred: {len(test_pred)}')
print(f'test_true: {len(test_true)}')

# %% evaluation

auc_value = accuracy_score(y_true=train_true, y_pred=train_pred)
print_('Accuracy value is %f for training dataset %s' % (auc_value, dataset))
print_(precision_recall_fscore_support(train_true, train_pred,
       average=None, labels=np.unique(train_true)))
print_(confusion_matrix(train_true, train_pred))

auc_value = accuracy_score(y_true=test_true, y_pred=test_pred)
print_('Accuracy value is %f for testing dataset %s' % (auc_value, dataset))
print_(precision_recall_fscore_support(
    test_true, test_pred, average=None, labels=np.unique(test_true)))
print_(confusion_matrix(test_true, test_pred))

exec_time = t2 - t1
print_('Execution time is %d seconds' % exec_time)
print_('No. of drifts is %d' % len(drifts_detected))


# %% plots

print_('plotting...')
plot_orbit(df_train, breaks_train, 'train_true', draw=[1, 2, 3, 4])
plot_orbit(df_train, breaks_train, 'train_pred', draw=[1, 2, 3, 4], labels=train_pred)
plot_orbit(df_test, breaks_test, 'test_true', draw=[1, 2, 3, 4])
plot_orbit(df_test, breaks_test, 'test_pred', draw=[1, 2, 3, 4], labels=test_pred)
print_('plotting finished')


# %% close log file

if fptr is not None:
    fptr.close()
    fptr = None

# %%
