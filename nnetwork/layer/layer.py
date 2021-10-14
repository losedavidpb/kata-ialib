import numpy as np
import abc

# ________________________ Layer definition ________________________

class Layer(abc.ABC):

    def __init__(self):
        self.__weights = None

    @abc.abstractmethod
    def predict(self, p_x, is_trained=True):
        pass

    @abc.abstractmethod
    def init_weights(self, n_size=1):
        pass

    @abc.abstractmethod
    def net_input(self, p_x, is_trained=True):
        pass

    @abc.abstractmethod
    def activation_function(self, p_net_input):
        pass

    @abc.abstractmethod
    def quantization(self, p_activation):
        pass

# _______________________ Layer types _______________________

class InputLayer(Layer, abc.ABC):

    def __init__(self, n_neurons=1):
        super().__init__()
        self.__n_neurons = n_neurons

    def init_weights(self, n_size=1):
        self.__weights = np.concatenate((
            np.zeros(shape=(1, self.__n_neurons)),
            np.eye(self.__n_neurons)))

class HiddenLayer(Layer, abc.ABC):

    def __init__(self, n_neurons=1, n_inputs_each_neuron=1):
        super().__init__()
        self.__n_neurons = n_neurons
        self.__n_inputs_each_neuron = n_inputs_each_neuron

    def init_weights(self, n_size=1):
        self.__weights = np.random.random(
            size=(1 + self.__n_inputs_each_neuron, self.__n_neurons))

class OutputLayer(Layer, abc.ABC):

    def __init__(self, n_neurons=1, n_inputs_each_neuron=1):
        super().__init__()
        self.__n_neurons = n_neurons
        self.__n_inputs_each_neuron = n_inputs_each_neuron

    def init_weights(self, n_size=1):
        self.__weights = np.random.random(
            size=(1 + self.__n_inputs_each_neuron, self.__n_neurons))
