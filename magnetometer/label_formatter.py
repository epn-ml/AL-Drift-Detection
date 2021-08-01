# %% imports

import pandas as pd

# %% loading the labels

path = '../data/'
labels_new_raw = pd.read_csv(path + 'jgra55678-sup-0002-table_si-s01-ts.csv')

# %% set the right type

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

labels_new.to_csv(path + 'messenger-0011_-4104_labelled.csv', index=False)
