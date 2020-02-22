import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.keras as keras

red = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
white = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
                    sep=';')
# print(red.head())
# print(white.head())

red['type'] = 0
white['type'] = 1

print(red.head(2))
print(white.head(2))

wine = pd.concat([red, white])
print(red.describe())
print(white.describe())
print(wine.describe())

plt.hist(wine['type'])
plt.xticks([0, 1])
plt.show()

print(wine['type'].value_counts())
print('*' * 30)
print('wine_norm')
wine_norm = (wine - wine.min()) / (wine.max() - wine.min())
print(wine_norm.head())
print(wine_norm.describe())

wine_shuffle = wine_norm.sample(frac=1)
print(wine_shuffle.head())
wine_np = wine_shuffle.to_numpy()
print(wine_np[:5])

train_idx = int(len(wine_np) * 0.8)
train_x, train_y = wine_np[:train_idx, :-1], wine_np[:train_idx, -1]
test_x, test_y = wine_np[train_idx:, :-1], wine_np[train_idx:, -1]
print(train_x[0])
print(train_y[0])
print(test_x[0])
print(test_y[0])
train_y = keras.utils.to_categorical(train_y, num_classes=2)
test_y = keras.utils.to_categorical(test_y, num_classes=2)
print(train_y[0])
print(test_y[0])

model = keras.Sequential([
    keras.layers.Dense(units=48, activation='relu', input_shape=(12,)),
    keras.layers.Dense(units=24, activation='relu'),
    keras.layers.Dense(units=12, activation='relu'),
    keras.layers.Dense(units=2, activation='softmax'),
])

model.compile(optimizer=keras.optimizers.Adam(lr=0.07),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
# history = model.fit(train_x, train_y, epochs=25, batch_size=32, validation_split=0.25)


x = np.arange(-2, 2, 0.01)
e_x = math.e ** x

plt.axhline(0, color='gray')
plt.axvline(0, color='gray')
plt.plot(x, x, 'b-', label='y=x')
plt.plot(x, e_x, 'g.', label='y=e^x')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

history = model.fit(train_x, train_y, epochs=25, batch_size=32, validation_split=0.25)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'g-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
plt.xlabel('Epoch')
plt.ylim(0.7, 1)
plt.legend()

plt.show()

print('*' * 10)
model.evaluate(test_x, test_y)

wine.loc[wine['quality'] <= 5, 'new_quality'] = 0
wine.loc[wine['quality'] == 6, 'new_quality'] = 1
wine.loc[wine['quality'] >= 7, 'new_quality'] = 2

print(wine['new_quality'].describe())
print(wine['new_quality'].value_counts())

del wine['quality']
wine_norm = (wine - wine.min()) / (wine.max() - wine.min())
wine_shuffle = wine_norm.sample(frac=1)
wine_np = wine_shuffle.to_numpy()

train_idx = int(len(wine_np) * 0.8)
train_X, train_Y = wine_np[:train_idx, :-1], wine_np[:train_idx, -1]
test_X, test_Y = wine_np[train_idx:, :-1], wine_np[train_idx:, -1]
train_Y = keras.utils.to_categorical(train_Y, num_classes=3)
test_Y = keras.utils.to_categorical(test_Y, num_classes=3)

model = keras.Sequential([
    keras.layers.Dense(units=48, activation='relu', input_shape=(12,)),
    keras.layers.Dense(units=24, activation='relu'),
    keras.layers.Dense(units=12, activation='relu'),
    keras.layers.Dense(units=3, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(lr=0.07), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_x, train_y, epochs=25, batch_size=32, validation_split=0.25)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'g-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
plt.xlabel('Epoch')
plt.ylim(0.7, 1)
plt.legend()

plt.show()
