import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt

class NN():
    def __init__(self, X, y, n_input, n_output, n_hidden, lr, batch_size, max_iter):
        self.X = X
        self.y = y
        self.n_input = n_input
        self.n_output = n_output
        self.n_h = n_hidden
        self.lr = lr
        self.batch_size = batch_size
        self.max_iter = max_iter

    def init_param(self):
        W1 = np.random.randn(self.n_input, self.n_h) / np.sqrt(self.n_input)
        b1 = np.zeros(self.n_h)
        W2 = np.random.randn(self.n_h, self.n_output) / np.sqrt(self.n_output)
        b2 = np.zeros(self.n_output)

        param = {'W1': W1,
                 'b1': b1,
                 'W2': W2,
                 'b2': b2}
        return param

    def forward_propagation(self, batch_x, param):
        W1, b1, W2, b2 = param['W1'], param['b1'], param['W2'], param['b2']

        Z1 = np.dot(batch_x, W1) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = np.exp(Z2 - np.max(Z2, axis=1, keepdims=True))
        A2 /= np.sum(A2, axis=1, keepdims=True)

        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": A2}
        return A2, cache

    def cost(self, A2, batch_y):
        m = batch_y.shape[0]
        loss = -np.sum(np.log(A2[np.arange(m), batch_y])) / m
        return loss

    def back_propagation(self, batch_x, batch_y, param, cache):
        m = batch_y.shape[0]
        W1, W2 = param['W1'], param['W2']
        A1, A2 = cache['A1'], cache['A2']

        dZ2 = A2.copy()
        dZ2[np.arange(m), batch_y] -= 1
        dW2 = 1. / m * np.dot(A1.T, dZ2)
        db2 = 1. / m * np.sum(dZ2, axis=0)
        dZ1 = np.dot(dZ2, W2.T) * (1-np.power(A1,2))
        dW1 = 1. / m * np.dot(batch_x.T, dZ1)
        db1 = 1. / m * np.sum(dZ1, axis=0)
        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}
        return grads

    def update_param(self, param, grads):
        W1, b1, W2, b2 = param['W1'], param['b1'], param['W2'], param['b2']
        dW1, db1, dW2, db2 = grads['dW1'], grads['db1'], grads['dW2'], grads['db2']

        W1 -= self.lr * dW1
        b1 -= self.lr * db1
        W2 -= self.lr * dW2
        b2 -= self.lr * db2
        param = {'W1': W1,
                 'b1': b1,
                 'W2': W2,
                 'b2': b2}
        return param

    def optimize(self, print_loss=False):
        num_samples, n_x = self.X.shape
        n_y = self.batch_size
        param = self.init_param()

        for i in range(self.max_iter):
            batch_idxs = np.random.choice(num_samples, self.batch_size)
            batch_x = self.X[batch_idxs]
            batch_y = self.y[batch_idxs]

            # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
            A2, cache = self.forward_propagation(batch_x, param)

            # Cost function. Inputs: "A2, Y, parameters". Outputs: "loss".
            loss = self.cost(A2, batch_y)

            # Backpropagation. Inputs: "X, Y, parameters, cache". Outputs: "grads".
            grads = self.back_propagation(batch_x, batch_y, param, cache)

            # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
            param = self.update_param(param, grads)

            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, loss))


if __name__=="__main__":

    # 生成数据集
    np.random.seed(0)
    X, y = sklearn.datasets.make_moons(200, noise=0.2)

    lr = 0.01
    reg_lambda = 0.01
    n_input = X.shape[1]
    n_output = 2
    n_hidden = 10

    # plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    # plt.show()

    model = NN(X=X,
               y=y,
               n_input=n_input,
               n_output=n_output,
               n_hidden=n_hidden,
               lr=lr,
               batch_size=200,
               max_iter=50000)

    model.optimize(print_loss=True)

