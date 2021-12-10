from nnetwork.neurons.neuron import Neuron
import numpy as np

"""
    Adaline (Adaptine Linear Element) is a binary classifier created by
    Bernand Widrow and Ted Hoff in 1960. It is inspired by McCulloch-Pitts
    computational model of a neuron.

    Adaline's computational model can be divided into three sections:

        * Net input function: process that returns a result depending on
        the values of inputs and weights. This section generally is described
        as the aggregation of the product of each weight and input.

        * Activation function: makes the decision based on the result returned
        by the net input function. Adaline neurons implements the activation
        function so it return the same as the net input function.

        * Thresholding function (quantization): takes the value returned by the
        activation function and changes it in order to get what is the class
        predicted for input passed to the neuron.

    This neuron is an improvement of the Perceptron model, since it uses continuous
    predicted values (from the net input) to learn the model coefficients, which is
    more “powerful” since it tells us by “how much” we were right or wrong.

    IMPORTANT!! Adaline is just for one neuron. Madaline is the model
    that includes many Adaline neurons (neural network).
"""

class AdalineGD(Neuron):
    """ Adaline model that uses gradient descent. """

    def __init__(self, lr=0.01, n_epochs=10):
        self.__history = {'weights': [], 'errors': []}
        self.__best_weights = None
        self.__n_epochs = n_epochs
        self.__lr = lr

    def best_weights(self):
        return self.__best_weights

    def fit(self, p_x, p_y, init_weights=True, max_tries=1):
        self.__history.clear()
        self.__history = {'weights': [], 'errors': []}
        if init_weights is True: self._init_weights(p_x)

        # Best weights will have the minimum cost value
        self.__best_weights = self.__history['weights'][0]

        # Number of epochs on which weights have gotten worse.
        # This value will reset weights whether they improve!
        num_tries = 0

        for _ in range(0, self.__n_epochs):
            v_errors = p_y - self._activation_function(self._net_input(p_x, is_trained=False))
            self._update_weights(p_x, v_errors)
            self.__history['errors'].append(self._error(v_errors))

            if len(self.__history['errors']) >= 2:
                prev_error = self.__history['errors'][len(self.__history['errors']) - 2]
                last_error = self.__history['errors'][len(self.__history['errors']) - 1]

                if prev_error <= last_error:
                    num_tries = num_tries + 1

                    if num_tries >= max_tries:
                        self._best_weight()
                        break

        self._best_weight()
        return self.__history

    def predict(self, p_x, is_trained=True):
        # First weight is the thresholding parameter (bias)
        return self._quantization(self._activation_function(self._net_input(p_x, is_trained)))

    def _init_weights(self, p_x):
        self.__history['weights'].clear()
        self.__history['weights'].append(np.random.random(size=p_x.shape[1] + 1))

    def _update_weights(self, p_x, v_errors):
        last_w = self.__history['weights'][len(self.__history['weights']) - 1]

        # Weights will be updated taking into account that the result is an array
        last_w[1:] = last_w[1:] + self.__lr * p_x.T.dot(v_errors)

        # Since x_0 = 1, it is necessary to update it differently. The operation
        # is similar than the rest of weights, but it is important to notice that
        # the result must be a value not an array
        last_w[0] = last_w[0] + self.__lr * v_errors.sum()

        self.__history['weights'].append(last_w)

    def _best_weight(self):
        index_min = min(range(len(self.__history['errors'])), key=self.__history['errors'].__getitem__)
        self.__best_weights = self.__history['weights'][index_min]

    @staticmethod
    def _error(v_errors):
        return 0.5 * (v_errors ** 2).sum()

    def _net_input(self, p_x, is_trained=True):
        last_w = self.__best_weights if is_trained else self.__history['weights'][len(self.__history['weights']) - 1]
        return np.dot(p_x, last_w[1:]) + last_w[0]

    @staticmethod
    def _activation_function(p_net_input):
        return p_net_input

    @staticmethod
    def _quantization(p_activate):
        # Quantization is represented as a binary prediction
        return np.where(p_activate >= 0.0, 1, -1)
