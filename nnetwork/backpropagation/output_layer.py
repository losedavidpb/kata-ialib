from nnetwork.backpropagation import layer
import numpy as np

class OutputLayer(layer.Layer):

    def init_weights(self):
        size_weights = (1 + self.num_inputs_each_neuron, self.num_neurons)
        self.weights = np.random.normal(loc=0.0, scale=1, size=size_weights)
        return self.weights

    def predict(self, p_x):
        self.last_net_input = self.net_input(p_x)
        self.last_output = self.activation(self.last_net_input)
        return self.quantization(self.last_output)
