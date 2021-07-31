# %% some imports

import glob

import numpy as np
import pandas as pd

# %% loading the labels

path = '../data/'
labels = pd.read_csv(path + 'messenger-0000_-0200_labelled.csv')
labels_new_raw = pd.read_csv(path + 'jgra55678-sup-0002-table_si-s01-ts.csv')

# %% setting datatype right

labels_date_list = ['SK outer in', 'SK inner in', 'MP outer in', 'MP inner in',
                    'MP inner out', 'MP outer out', 'SK inner out', 'SK outer out']

for date_feat in labels_date_list:
    labels[date_feat] = pd.to_datetime(labels[date_feat])

labels_new_raw['Timestamp'] = pd.to_datetime(labels_new_raw['Timestamp'])

# %% reformat new labels


def findOrbit(data, o):
    index = [i for i, row in enumerate(data) if row[0] == o]

    if len(index) > 0:
        return index[0]

    return -1


labels_new_data = list()

for _, row in labels_new_raw.iterrows():
    o = row['Orbit number']
    b = row['Boundary number']
    t = row['Timestamp']
    i = findOrbit(labels_new_data, o)

    if i == -1:
        labels_new_data.append(
            [o, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT])

    labels_new_data[i][b] = t

labels_new = pd.DataFrame(labels_new_data,
                          columns=['Orbit', 'SK outer in', 'SK inner in', 'MP outer in', 'MP inner in',
                                   'MP inner out', 'MP outer out', 'SK inner out', 'SK outer out', 9, 10])

# %% loading the training data (50 orbits)

all_files = glob.glob(path + 'orbits/*.csv')

li = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

df_train_init = pd.concat(li, axis=0, ignore_index=True)
df_train = df_train_init.dropna()

# %% change the datatype of the 'DATE' feature

df_train['DATE'] = pd.to_datetime(df_train['DATE'])

# %% transform the training data to a dataframe which contains the mean value over 1 minute

df_train_minute = df_train.copy()
df_train_minute.index = df_train_minute['DATE']
df_train_minute = df_train_minute.resample('1Min').mean()
df_train_minute = df_train_minute.reset_index()

# %% training data description

df_train_description = df_train.describe()
df_train_description.to_excel('df_train_description.xlsx')

df_train_minute_description = df_train_minute.describe()
df_train_minute_description.to_excel('df_train_minute_description.xlsx')

# %% assign labels to the instances of the training data


def labeller(df_train, labels):

    df_train['labels'] = 'IMF'

    for i in range(0, len(labels)):

        # bow shock crossing
        df_train.loc[((df_train['DATE'] > labels.iloc[i]['SK outer in']) & (df_train['DATE'] < labels.iloc[i]['SK inner in'])) |
                     ((df_train['DATE'] > labels.iloc[i]['SK inner out']) & (df_train['DATE'] < labels.iloc[i]['SK outer out'])), 'labels'] = 'BS-crossing'

        # magnetosheath
        df_train.loc[((df_train['DATE'] > labels.iloc[i]['SK inner in']) & (df_train['DATE'] < labels.iloc[i]['MP outer in'])) |
                     ((df_train['DATE'] > labels.iloc[i]['MP outer out']) & (df_train['DATE'] < labels.iloc[i]['SK inner out'])), 'labels'] = 'magnetosheath'

        # magnetosphere
        df_train.loc[(df_train['DATE'] > labels.iloc[i]['MP inner in']) & (
            df_train['DATE'] < labels.iloc[i]['MP inner out']), 'labels'] = 'magnetosphere'

        # magnetopause crossing
        df_train.loc[((df_train['DATE'] > labels.iloc[i]['MP outer in']) & (df_train['DATE'] < labels.iloc[i]['MP inner in'])) |
                     ((df_train['DATE'] > labels.iloc[i]['MP inner out']) & (df_train['DATE'] < labels.iloc[i]['MP outer out'])), 'labels'] = 'MP-crossing'

    return df_train


df_train_labelled = labeller(df_train, labels_new)
df_train_minute_labelled = labeller(df_train_minute, labels_new)

# %% df_train to_csv

df_train_labelled.to_csv('df_train_labelled.csv')
df_train_minute_labelled.to_csv('df_train_minute_labelled.csv')

# %% description of the labelled training data

df_train_labelled = df_train_labelled.drop(['DATE'], axis=1)
df_train_labelled_description = df_train_labelled.groupby(
    ['labels']).describe(include='all')
df_train_labelled_description.to_excel('df_train_labelled_description.xlsx')

df_train_minute_labelled = df_train_minute_labelled.drop(['DATE'], axis=1)
df_train_minute_labelled_description = df_train_minute_labelled.groupby(
    ['labels']).describe(include='all')
df_train_minute_labelled_description.to_excel(
    'df_train_minute_labelled_description.xlsx')

# %% split data into chunks and shuffle them


def split_df(df, window_size):

    chunks = list()
    num_chunks = len(df) // window_size + (1 if len(df) % window_size else 0)

    for i in range(num_chunks):
        chunks.append(df[i*window_size:(i+1)*window_size])

    return chunks


df_train_split = split_df(df_train, 10)
np.random.shuffle(df_train_split)

# %%
