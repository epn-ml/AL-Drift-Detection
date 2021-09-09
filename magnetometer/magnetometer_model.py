# %% imports

import glob

import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

# %% load data


def load_data(path):

    files = glob.glob(path)
    li = []

    for filename in files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)

    return df.dropna()


df_train = load_data('../data/labelled_orbits/train/*.csv')
df_test = load_data('../data/labelled_orbits/test/*.csv')


# %% select features


def select_features(df, features):

    drop_col = ['Unnamed: 0', 'DATE', 'X_MSO', 'Y_MSO', 'Z_MSO', 'BX_MSO', 'BY_MSO', 'BZ_MSO', 'DBX_MSO', 'DBY_MSO', 'DBZ_MSO', 'RHO_DIPOLE', 'PHI_DIPOLE', 'THETA_DIPOLE',
                'BABS_DIPOLE', 'BX_DIPOLE', 'BY_DIPOLE', 'BZ_DIPOLE', 'RHO', 'RXY', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'VABS', 'D', 'COSALPHA', 'EXTREMA', 'ORBIT']

    for feature in features:
        if feature in drop_col:
            drop_col.remove(feature)

    if 'INDEX' in features:
        drop_col.remove('Unnamed: 0')

    return df.drop(drop_col, axis=1)


features = ['X', 'Y', 'Z', 'COSALPHA']
df_train = select_features(df_train, features)
df_test = select_features(df_test, features)

# %% normalize data

df_train[features] = df_train[features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df_test[features] = df_test[features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# %% truncate data

# df_train = df_train[:10**(len(str(len(df_train)))-1)]
# df_test = df_test[:10**(len(str(len(df_test)))-1)]

# %% split input and output data

x_train = df_train.drop(['LABEL'], axis=1).values.reshape(-1, len(features), 1)
x_test = df_test.drop(['LABEL'], axis=1).values.reshape(-1, len(features), 1)
y_train = df_train[['LABEL']].values.reshape(-1)
y_test = df_test[['LABEL']].values.reshape(-1)

# %% set up the model

model = keras.Sequential()
model.add(layers.Conv1D(64, 2, activation='relu',
          input_shape=x_train.shape[1:]))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.MaxPooling1D())
model.add(layers.Flatten())
model.add(layers.Dense(5, activation='softmax'))

# %% fit and evaluate the model

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=10, verbose=2)
acc = model.evaluate(x_test, y_test, verbose=2)
print('Loss:', acc[0], ' Accuracy:', acc[1])

# %%
