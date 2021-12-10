from nnetwork.backpropagation import layer
import numpy as np

class InputLayer(layer.Layer):

    def __init__(self, activation, derivative_activation, num_neurons=1):
        layer.Layer.__init__(self, activation, derivative_activation, num_neurons, 1)

    def init_weights(self):
        zeros_num_neurons = np.zeros(1, self.num_neurons)
        eye_num_neurons = np.eye(self.num_neurons)
        self.weights = np.concatenate(zeros_num_neurons, eye_num_neurons)
        return self.weights

    def predict(self, p_x):
        self.last_net_input = self.net_input(p_x)
        self.last_output = self.activation(self.last_net_input)
        return self.last_output
