import abc

class Neuron(abc.ABC):
    """ Generic Neuron definition """

    @abc.abstractmethod
    def fit(self, p_x, p_y, init_weights=True, max_tries=1):
        """ Execute training process for passed input values

        :param p_x Numpy array of input values
        :param p_y List of class values for each characteristic vector
        :param init_weights Check if weights needs first to be initialized
        :param max_tries Maximum tries that the neuron has to improve its error
        """
        pass

    @abc.abstractmethod
    def predict(self, p_x):
        """ Execute prediction process for passed input values

        :param p_x: Numpy array of input values
        """
        pass
