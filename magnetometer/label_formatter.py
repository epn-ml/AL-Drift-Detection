# %% imports

import pandas as pd

# %% loading the labels

path = '../data/'
labels_new_raw = pd.read_csv(path + 'jgra55678-sup-0002-table_si-s01-ts.csv')

# %% set the right type

labels_new_raw['Timestamp'] = pd.to_datetime(labels_new_raw['Timestamp'])

# %% reformat new labels

labels_new_data = dict()
offset = -11

for _, row in labels_new_raw.iterrows():
    o = row['Orbit number'] + offset
    b = row['Boundary number'] - 1
    t = row['Timestamp']

    if o == 949:
        offset -= 8

    if o == 1024:
        offset -= 1

    if not o in labels_new_data:
        labels_new_data[o] = [pd.NaT] * 8

    # boundaries 9 and 10 are ignored
    if b < 8:
        labels_new_data[o][b] = t.strftime('%Y-%m-%d %X')

labels_new = pd.DataFrame.from_dict(labels_new_data, orient='index',
                                    columns=['SK outer in', 'SK inner in', 'MP outer in', 'MP inner in',
                                             'MP inner out', 'MP outer out', 'SK inner out', 'SK outer out'])
labels_new.index.name = 'Orbit'
labels_new.to_csv(path + 'messenger-4084_labelled.csv')
