import os
import torch
import tensorflow as tf
import numpy as np
import pandas as pd


class Magnetometer_data(tf.data.Dataset):

    def __init__(self, csv_file, features, sample_rate=1, transform=None, root=''):
        """Args:
        root (string): Base path
        features (list): selected features
        csv_file (string): Path to csv file with sensor data
        transform (callable, optional): Transforms to be applied.
        """
        assert len(features) % 3 == 0
        self.data_path = os.path.join(root, csv_file)
        self.feature_cols = features
        self.sample_rate = sample_rate
        self.data = pd.read_csv(self.data_path, usecols=self.feature_cols)
        # self.data = self.__normalise__(self.data)
        self.transform = transform
        try:
            self.labels = pd.read_csv(self.data_path, usecols=['labels'])
        except ValueError as e:
            self.labels = None

    def __len__(self):
        return len(self.data)

    def __label_to_int(self, label):
        label_ord = {'IMF': 0, 'BS-crossing': 1, 'MP-crossing': 2,
                     'magnetosheath': 3, 'magnetosphere': 4}
        return label_ord[label]

    def __normalise__(self, features):
        mean = np.mean(features)
        std = np.std(features)
        return (features - mean)/std

    def __reshape__(self, x):
        # reshape bands, features
        return x.reshape(-1, 3)
        # return torch.unsqueeze(x,0)

    def __getitem__(self, idx):
        coords = self.__reshape__(torch.tensor(self.data.iloc[idx].values))
        if self.transform:
            coords = self.transform(coords)

        if self.labels is not None:
            label_int = torch.tensor(self.__label_to_int(
                self.labels.iloc[idx]['labels']))
            sample = {'coords': coords, 'labels': label_int}
            return sample
        else:
            sample = {'coords': coords}
            return sample
