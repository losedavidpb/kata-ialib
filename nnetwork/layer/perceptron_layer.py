from nnetwork.layer.layer import InputLayer, Layer, HiddenLayer, OutputLayer
import numpy as np
import abc

# _______________________ Madaline layer definition _______________________

class LayerPerceptron(Layer, abc.ABC):

    def net_input(self, p_x, is_trained=True):
        return np.dot(p_x, super().__weights[:1, :]) + super().__weights[0, :]

    def activation_function(self, p_net_input):
        return 1 / (1 + np.exp(-1 * p_net_input))

    def quantization(self, p_activation):
        return p_activation

# ________________ Madaline layer types ________________

class InputLayerPerceptron(LayerPerceptron, InputLayer):

    def predict(self, p_x, is_trained=True):
        return self.net_input(p_x, is_trained)

class HiddenLayerPerceptron(LayerPerceptron, HiddenLayer):

    def predict(self, p_x, is_trained=True):
        return self.activation_function(self.net_input(p_x, is_trained))

class OutputLayerPerceptron(LayerPerceptron, OutputLayer):

    def predict(self, p_x, is_trained=True):
        return self.activation_function(self.net_input(p_x, is_trained))
