from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):

    def __init__(self, activation, derivative_activation, num_neurons=1, num_inputs_each_neuron=1):
        self.num_neurons, self.num_inputs_each_neuron = num_neurons, num_inputs_each_neuron
        self.activation_f = {'activation': activation, 'derivative_activation': derivative_activation}
        self.last_net_input, self.last_output = 0, 0
        self.weights = None

    def net_input(self, p_x):
        return np.matmul(p_x, self.weights[1:, :]) + self.weights[0, :]

    def activation(self, net_input):
        return self.activation_f['activation'](net_input)

    def derivative_activation(self):
        return self.activation_f['derivative_activation'](self.last_net_input)

    @staticmethod
    def quantization(p_activation):
        return np.where(p_activation >= 0.5, 1, 0)

    @abstractmethod
    def init_weights(self):
        """ Initialize random weights for current layer """
        pass

    @abstractmethod
    def predict(self, p_x):
        """ Execute prediction process for passed input values

        :param p_x: array of input values
        """
        pass
