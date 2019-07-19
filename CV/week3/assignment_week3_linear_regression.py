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

class LinearRegression(object):
    def __init__(self, X, y, lr, batch_size, max_iter):
        self.X = X
        self.y = y
        self.lr = lr
        self.batch_size = batch_size
        self.max_iter = max_iter

    def inference(self, X, w, b):
        return np.dot(X, w) + b

    def loss(self, y_pred, y):
        grad = y_pred - y
        loss = np.dot(grad.T, grad) / (2 * self.batch_size)
        return loss

    def cal_step_gradient(self, x, y, y_pred, w, b):
        grad = y_pred - y

        dw = np.sum(np.dot(grad.T, x), axis=0) / self.batch_size
        db = (1.0 / self.batch_size) * np.sum(grad)
        w -= self.lr * dw
        b -= self.lr * db
        return w, b

    def train(self):
        num_samples, num_features = self.X.shape
        w = np.zeros(num_features)
        b = 0
        for i in range(self.max_iter):
            batch_idxs = np.random.choice(num_samples, self.batch_size)
            batch_x = self.X[batch_idxs]
            batch_y = self.y[batch_idxs]

            y_pred = self.inference(batch_x, w, b)

            w, b = self.cal_step_gradient(batch_x, batch_y, y_pred, w, b)
            loss = self.loss(y_pred, batch_y)
            print('iteration: {0}, w:{1}, b:{2}, loss is {3}'.format(i, w, b, loss))

        y_pred = np.dot(self.X, w) + b
        plt.plot(self.X, y_pred, 'r', label='Prediction')
        plt.scatter(self.X, self.y, marker=".", label="Training Data")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend(loc=2)
        plt.show()

    def run(self):
        self.train()


def gen_sample_data():
    num_samples = 200
    num_features = 1
    X = np.random.rand(num_samples, num_features) * 10
    w = np.random.rand(num_features)  # for noise random.random[0, 1)
    b = 2
    y = np.dot(X, w) + b + np.random.randn(num_samples, ) * 0.2
    return X, y


if __name__ == '__main__':
    X, y = gen_sample_data()
    lr = LinearRegression(X, y, lr=0.001, batch_size=50, max_iter=10000)
    lr.run()
