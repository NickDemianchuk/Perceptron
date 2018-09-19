import numpy as np

class Perceptron(object):

    def __init__(self, lrn_rate=0.1, epochs=100):
        """ Perceptron classifier

        Parameters
        ------------
        lrn_rate : float
            Learning rate - eta (between 0.0 and 1.0)
        epochs : int
            Iterations over the training dataset

        Attributes
        -----------
        w: 1-d array
            A weights vector
        errors: list
            A list of misclassifications per epoch
        """
        self.lrn_rate = lrn_rate
        self.epochs = epochs
        self.w = None
        self.errors = None

    def net_input(self, sample):
        """
        Parameters
        ------------
        sample : 1-d array
            One of the samples from dataset
        bias : float
            A weight-zero

        Returns
        ------------
        float
            A vector dot product (wTx)
        """
        bias = self.w[0]
        return np.dot(sample, self.w[1:]) + bias

    def activation(self, sample):
        """
        Parameters
        ------------
        sample : 1-d array
            One of the samples from dataset

        Returns
        ------------
        int
            A predicted class label (-1; 1) based on the net input result
        """
        return 1 if self.net_input(sample) >= 0 else -1

    def fit(self, X, y):
        """ Fit training data.
        Parameters
        ------------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        errors_i : int
            A number of errors in i-th iteration
        w_update : float
            A weights update value
        target: int
            Actual class label
        sample: 1-d array
            One of the samples from dataset

        Returns
        ------------
        self : object
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
