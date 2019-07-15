# -*- coding: utf-8 -*-

## Linear Regression
###############################
import numpy as np
import random

# X: (100, 1)
# y: (100, 1)
# w: (1)
# b: (1)

def cal_step_gradient(batch_x, batch_y, w, b, lr):
    m = batch_x.shape[0]
    y_pred = np.dot(batch_x, w) + b
    diff = y_pred - batch_y
    loss = np.dot(diff.T, diff) / (2 * m)

    dw = np.dot(diff.T, batch_x)/m
    db = (1.0 / m) * np.sum(diff)
    w -= lr * dw
    b -= lr * db
    return loss, w, b


def train(X, y, batch_size, lr, max_iter):
    w = random.randint(0, 10)
    b = random.randint(0, 5)
    num_samples = X.shape[0]
    for i in range(max_iter):
        batch_idxs = np.random.choice(num_samples, batch_size)
        batch_x = X[batch_idxs]
        batch_y = y[batch_idxs]
        loss, w, b = cal_step_gradient(batch_x, batch_y, w, b, lr)
        print('iteration: {0}, w:{1}, b:{2}, loss is {3}'.format(i, w, b, loss))


def gen_sample_data():
    w = random.randint(0, 10) + random.random()		# for noise random.random[0, 1)
    b = random.randint(0, 5) + random.random()
    num_samples = 100
    X = np.random.randint(0, 100, (num_samples, 1)) * random.random()
    y = np.dot(X, w) + b + random.random() * random.randint(-1, 1)
    return X, y, w, b


def run():
    X, y, w, b = gen_sample_data()
    lr = 0.001
    max_iter = 10000
    train(X, y, 50, lr, max_iter)


if __name__ == '__main__':
    run()
