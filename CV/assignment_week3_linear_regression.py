# -*- coding: utf-8 -*-

## Linear Regression
###############################
import numpy as np
import random
import matplotlib.pyplot as plt

# X: (100, 1)
# y: (100, 1)
# w: (1)
# b: (1)

def cal_step_gradient(batch_x, batch_y, w, b, lr):
    m = batch_x.shape[0]
    y_pred = np.dot(batch_x, w) + b
    diff = y_pred - batch_y
    loss = np.dot(diff.T, diff) / (2 * m)

    dw = np.sum(np.dot(diff.T, batch_x), axis=0)/m
    db = (1.0 / m) * np.sum(diff)
    w -= lr * dw
    b -= lr * db
    return loss, w, b


def train(X, y, batch_size, lr, max_iter):
    w = np.zeros(X.shape[1])
    b = 0
    num_samples = X.shape[0]
    print("w shape = ", w.shape)
    for i in range(max_iter):
        batch_idxs = np.random.choice(num_samples, batch_size)
        batch_x = X[batch_idxs]
        batch_y = y[batch_idxs]
        loss, w, b = cal_step_gradient(batch_x, batch_y, w, b, lr)
        print('iteration: {0}, w:{1}, b:{2}, loss is {3}'.format(i, w, b, loss))

    y_pred = np.dot(X, w) + b
    plt.plot(X, y_pred, 'r', label='Prediction')
    plt.scatter(X, y, marker=".", label="Training Data")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend(loc=2)
    plt.show()


def gen_sample_data():
    num_samples = 100
    X = np.random.rand(num_samples, 1) * 10
    w = np.random.rand(X.shape[1]) 		# for noise random.random[0, 1)
    b = 2
    y = np.dot(X, w) + b + np.random.randn(num_samples,)* 0.2
    return X, y, w, b


def run():
    X, y, w, b = gen_sample_data()
    lr = 0.001
    max_iter = 10000
    train(X, y, 50, lr, max_iter)


if __name__ == '__main__':
    run()
