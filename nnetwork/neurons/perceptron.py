import numpy as np
from nnetwork.neurons.neuron import Neuron

class PerceptronGD(Neuron):
    """ Perceptron model that uses gradient descent. """

    def __init__(self, lr=0.01, n_epochs=10):
        self.__weights, self.__errors = [], []
        self.__best_weights = None
        self.__n_epochs = n_epochs
        self.__lr = lr

    def best_weights(self):
        return self.__best_weights

    def fit(self, p_x, p_y, init_weights=True, max_tries=1):
        self.__weights, self.__errors = [], []
        if init_weights is True: self._init_weights(p_x)

        # Best weights will have the minimum cost value
        self.__best_weights = self.__weights[0]

        # Number of epochs on which weights have gotten worse.
        # This value will reset weights whether they improve!
        num_tries = 0

        for _ in range(0, self.__n_epochs):
            v_errors = p_y - self.predict(p_x, is_trained=False)
            self._update_weights(p_x, v_errors)
            self.__errors.append(self._error(v_errors))

            if len(self.__errors) >= 2:
                prev_error = self.__errors[len(self.__errors) - 2]
                last_error = self.__errors[len(self.__errors) - 1]

                if prev_error <= last_error:
                    num_tries = num_tries + 1

                    if num_tries >= max_tries:
                        self._best_weight()
                        break

        return self.__weights, self.__errors

    def predict(self, p_x, is_trained=True):
        return self._activation_function(self._net_input(p_x, is_trained))

    def _best_weight(self):
        index_min = min(range(len(self.__errors)), key=self.__errors.__getitem__)
        self.__best_weights = self.__weights[index_min]

    def _init_weights(self, p_x):
        self.__weights.append(np.random.random(size=p_x.shape[1] + 1))

    def _update_weights(self, p_x, v_errors):
        last_w = self.__weights[len(self.__weights) - 1]

        # Weights will be updated taking into account that the result is an array
        last_w[1:] = last_w[1:] + self.__lr * p_x.T.dot(v_errors)

        # Since x_0 = 1, it is necessary to update it differently. The operation
        # is similar than the rest of weights, but it is important to notice that
        # the result must be a value not an array
        last_w[0] = last_w[0] + self.__lr * v_errors.sum()

        self.__weights.append(last_w)

    @staticmethod
    def _error(v_errors):
        return (v_errors ** 2).sum() / v_errors.shape[0]

    def _net_input(self, p_x, is_trained=True):
        last_w = self.__best_weights if is_trained else self.__weights[len(self.__weights) - 1]
        return np.dot(p_x, last_w[1:]) + last_w[0]

    @staticmethod
    def _activation_function(p_net_input):
        return 1 / (1 + np.exp(-1 * p_net_input))
