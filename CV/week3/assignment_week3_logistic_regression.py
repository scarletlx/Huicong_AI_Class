# -*- coding: utf-8 -*-

## Linear Regression
###############################
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn import preprocessing as pp


# X: (100, 3)
# y: (100, 1)
# w: (3)
# b: (1)

class LogisticRegression():
    def __init__(self, X, y, lr, batch_size, max_iter):
        self.X = X
        self.y = y
        self.lr = lr
        self.batch_size = batch_size
        self.max_iter = max_iter

    def sigmoid(self, z):
        return 1.0 / (1 + np.exp(-z))

    def inference(self, x, w):
        return self.sigmoid(np.dot(x, w))

    def cost(self, y_pred, y):
        loss = np.mean(np.multiply(-y, np.log(y_pred)) - np.multiply((1-y), np.log(1-y_pred)))
        return loss

    def gradient(self, grad, x, w):
        dw = np.dot(x.T, grad) / self.batch_size
        w += self.lr * dw
        return w

    def train(self):
        num_samples, num_features = self.X.shape
        w = np.zeros(num_features)


        for i in range(self.max_iter):
            batch_x = self.X[batch_idxs]
            batch_y = self.y[batch_idxs]

            y_pred = self.inference(batch_x, w)
            grad = batch_y - y_pred
            loss = self.cost(y_pred, batch_y)
            w = self.gradient(grad, batch_x, w)
            print('iteration: {0}, w:{1}, loss is {2}'.format(i, w, loss))
        batch_idxs = np.random.choice(num_samples, self.batch_size)

        positive = self.X[self.y == 1]
        negative = self.X[self.y == 0]

        plt.scatter(positive[:,1], positive[:,2], c="g", marker=".", label="Accepted")
        plt.scatter(negative[:,1], negative[:,2], c="r", marker="x", label="Rejected")

        x_min = min(self.X[:,1])
        x_max = max(self.X[:,1])
        draw_x = np.arange(x_min, x_max, 0.5)
        draw_y = (-w[0] - w[1] * draw_x) / w[2]
        plt.plot(draw_x, draw_y)

        plt.xlabel("Exam 1")
        plt.ylabel("Exam 2")
        plt.legend(loc=2)
        plt.show()

    def run(self):
        self.train()


def gen_sample_data():
    path = 'assignment_week3_logistic_regression_data.txt'
    data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
    data.insert(0, 'Ones', 1)
    cols = data.shape[1]
    orig_data = data.as_matrix()
    X = orig_data[:, :cols - 1]
    y = orig_data[:, cols-1]

    return X, y

if __name__ == '__main__':
    X, y = gen_sample_data()
    scaled_X = pp.scale(X)

    log_reg = LogisticRegression(scaled_X, y, lr=0.001, batch_size=16, max_iter=40000)
    log_reg.run()
