import numpy as np
import tensorflow as tf

x = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
y = np.array([[1], [1], [1], [0]])
w = tf.random.normal([2], 0, 1)
b = tf.random.normal([1], 0, 1)
b_x = 1

for i in range(2000):
    error_sum = 0
    for j in range(4):
        output = tf.sigmoid(np.sum(x[j] * w) + b_x * b)
        error = y[j][0] - output
        w = w + x[j] * 0.1 * error
        b = b + b_x * 0.1 * error
        error_sum += error

    if i % 200 == 199:
        print(f'{i + 1:5} : {error_sum}')

for i in range(4):
    print(f'X: {x[i]}, Y: {y[i]} Output: {tf.sigmoid(np.sum(x[i] * w) + b)}')
