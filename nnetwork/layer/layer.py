import numpy as np
import abc

# ________________________ Layer definition ________________________

class Layer(abc.ABC):
    """ Layer definition for the creation of a NN """

    def __init__(self):
        self.__weights = None

    @abc.abstractmethod
    def predict(self, p_x, is_trained=True):
        """ Execute prediction process for passed input values

        :param p_x: Numpy array of input values
        :param is_trained: Check if current layer is trained
        """
        pass

    @abc.abstractmethod
    def init_weights(self, n_size=1):
        """ Initialize weights of each neuron for current layer

        :param n_size: Number of weights/neuron for current layer
        """
        pass

    @abc.abstractmethod
    def net_input(self, p_x, is_trained=True):
        """ Net input for current layer

        :param p_x: Numpy array of input values
        :param is_trained: Check if current layer is trained
        """
        pass

    @abc.abstractmethod
    def activation_function(self, p_net_input):
        """ Activation function for current layer

        :param p_net_input: value returned by net_input
        """
        pass

    @abc.abstractmethod
    def quantization(self, p_activation):
        """ Quantization section for current layer

        :param p_activation: value returned by activation function
        """
        pass

# _______________________ Layer types _______________________

class InputLayer(Layer, abc.ABC):
    """ First layer of a NN that contains the inputs values """

    def __init__(self, n_neurons=1):
        super().__init__()
        self.__n_neurons = n_neurons

    def init_weights(self, n_size=1):
        self.__weights = np.concatenate((
            np.zeros(shape=(1, self.__n_neurons)),
            np.eye(self.__n_neurons)))

class HiddenLayer(Layer, abc.ABC):
    """ Layer of a NN that are before the InputLayer """

    def __init__(self, n_neurons=1, n_inputs_each_neuron=1):
        super().__init__()
        self.__n_neurons = n_neurons
        self.__n_inputs_each_neuron = n_inputs_each_neuron

    def init_weights(self, n_size=1):
        self.__weights = np.random.random(
            size=(1 + self.__n_inputs_each_neuron, self.__n_neurons))

class OutputLayer(Layer, abc.ABC):
    """ Last Layer of a NN that return the final output """

    def __init__(self, n_neurons=1, n_inputs_each_neuron=1):
        super().__init__()
        self.__n_neurons = n_neurons
        self.__n_inputs_each_neuron = n_inputs_each_neuron

    def init_weights(self, n_size=1):
        self.__weights = np.random.random(
            size=(1 + self.__n_inputs_each_neuron, self.__n_neurons))
