# %% imports

import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
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


df_train = load_data('../data/orbits/train/*.csv')
df_test = load_data('../data/orbits/test/*.csv')


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


features = ['X_MSO', 'Y_MSO', 'Z_MSO', 'BX_MSO', 'BY_MSO', 'BZ_MSO', 'DBX_MSO', 'DBY_MSO', 'DBZ_MSO',
            'RHO_DIPOLE', 'BX_DIPOLE', 'BY_DIPOLE', 'BZ_DIPOLE', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'COSALPHA', 'EXTREMA']
df_train = select_features(df_train, features)
df_test = select_features(df_test, features)

# %% normalize data

# df_train[features] = df_train[features].apply(
#     lambda x: (x - x.min()) / (x.max() - x.min()))
# df_test[features] = df_test[features].apply(
#     lambda x: (x - x.min()) / (x.max() - x.min()))

df_train[features] = df_train[features].apply(zscore)
df_test[features] = df_test[features].apply(zscore)

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
model.add(layers.Conv1D(64, 3, activation='relu',
          input_shape=x_train.shape[1:]))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.MaxPooling1D())
model.add(layers.Flatten())
model.add(layers.Dense(5, activation='softmax'))

# %% fit and evaluate the model

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
              optimizer='adam',
              metrics=['accuracy'])

weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train)

print(weights)

history = model.fit(x=x_train,
                    y=y_train,
                    batch_size=16,
                    epochs=20,
                    class_weight={k: v for k, v in enumerate(weights)},
                    verbose=2)
acc = model.evaluate(x_test, y_test, verbose=2)

# %% plot

plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %% output

print('Loss:', acc[0], ' Accuracy:', acc[1])
pred = model.predict(x_test)
pred_y = pred.argmax(axis=-1)
cm = confusion_matrix(y_test, pred_y)
print(cm)

# %% precision and recall


def calc_stats(cm):

    precision = []
    recall = []

    for i in range(len(cm)):
        tp = cm[i][i]
        fp = 0
        fn = 0

        for j in range(len(cm)):
            if j != i:
                fp += cm[j][i]
                fn += cm[i][j]

        precision.append(tp / (tp+fp))
        recall.append(tp / (tp + fn))

    print(precision)
    print(recall)


calc_stats(cm)

# %%
