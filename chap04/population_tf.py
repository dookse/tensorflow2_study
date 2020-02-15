import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Least Seuare Method

X = [0.3, -0.78, 1.26, 0.03, 1.11, 0.24, -0.24, -0.47, -0.77, -0.37, -0.85, -0.41, -0.27, 0.02, -0.76, 2.66]
Y = [12.27, 14.44, 11.87, 18.75, 17.52, 16.37, 19.78, 19.51, 12.65, 14.74, 10.72, 21.94, 12.83, 15.51, 17.14, 14.42]

a = tf.Variable(random.random())
b = tf.Variable(random.random())


def comput_loss():
    y_pred = a * X + b
    loss = tf.reduce_mean((Y - y_pred) ** 2)
    return loss


optimizer = tf.optimizers.Adam(lr=0.07)
for i in range(1000):
    optimizer.minimize(comput_loss, var_list=[a, b])

    if i % 100 == 99:
        print(f'{i + 1} a:{a.numpy()} b:{b.numpy()} loss:{comput_loss().numpy()}')

line_x = np.arange(min(X), max(X), 0.01)
line_y = a * line_x + b

plt.plot(line_x, line_y, 'r-')

plt.plot(X, Y, 'bo')
plt.xlabel('Population Growth Rate (%)')
plt.ylabel('Elderly Population Rate (%)')
plt.show()
