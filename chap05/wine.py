import matplotlib.pyplot as plt
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
test_x, test_y = wine_np[train_idx:, :-1], wine_np[train_idx:-1]
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
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
