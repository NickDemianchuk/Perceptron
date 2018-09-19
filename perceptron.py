import numpy as np

class Perceptron(object):

    def __init__(self, lrn_rate=0.1, epochs=100):
        """
        :param lrn_rate: Learning rate between 0.0 and 1 (eta)
        :param epochs: Number of iterations over the dataset
        :attribute w: A weights vector
        :attribute errors: An array of misclassifications per epoch
        """
        self.lrn_rate = lrn_rate
        self.epochs = epochs
        self.w = None
        self.errors = None

    def net_input(self, sample):
        """
        :param sample: One of the samples from dataset
        :var bias: A weight-zero
        :return: A vector dot product (wTx)
        """
        bias = self.w[0]
        return np.dot(sample, self.w[1:]) + bias

    def activation(self, sample):
        """
        :param sample: One of the samples from dataset
        :return: Predicted class label (-1; 1) based on the net input result
        """
        return 1 if self.net_input(sample) >= 0 else -1

    def fit(self, X, y):
        """
        :param X: Multidimensional array that represents the dataset
        :param y: A vector of target values
        :var errors_i: A number of errors in i-th iteration
        :var w_update: A weights update value
        :var target: Actual class label
        :var sample: One of the samples from dataset
        :return: self
        """
        self.errors = []
        self.w = np.zeros(X.shape[1] + 1)

        for _ in range(self.epochs):
            errors_i = 0
            for sample, target in zip(X, y):
                w_update = self.lrn_rate * (target - self.activation(sample))
                self.w[0] += w_update
                self.w[1:] += w_update * sample
                if w_update != 0:
                    errors_i += 1
            self.errors.append(errors_i)

        return self
