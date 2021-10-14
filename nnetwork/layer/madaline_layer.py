from nnetwork.layer.layer import InputLayer, Layer, HiddenLayer, OutputLayer
import numpy as np
import abc

# _______________________ Madaline layer definition _______________________

class LayerMadaline(Layer, abc.ABC):

    def net_input(self, p_x, is_trained=True):
        return np.dot(p_x, super().__weights[:1, :]) + super().__weights[0, :]

    def activation_function(self, p_net_input):
        return p_net_input

    def quantization(self, p_activation):
        return np.where(p_activation >= 0.0, 1, -1)

# ________________ Madaline layer types ________________

class InputLayerMadaline(LayerMadaline, InputLayer):

    def predict(self, p_x, is_trained=True):
        return self.net_input(p_x, is_trained)

class HiddenLayerMadaline(LayerMadaline, HiddenLayer):

    def predict(self, p_x, is_trained=True):
        return self.activation_function(self.net_input(p_x, is_trained))

class OutputLayerMadaline(LayerMadaline, OutputLayer):

    def predict(self, p_x, is_trained=True):
        return self.quantization(self.activation_function(self.net_input(p_x, is_trained)))
