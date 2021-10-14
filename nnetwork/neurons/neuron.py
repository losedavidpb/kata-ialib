import abc

class Neuron(abc.ABC):

    @abc.abstractmethod
    def fit(self, p_x, p_y, init_weights=True, max_tries=1):
        pass

    @abc.abstractmethod
    def predict(self, p_x):
        pass
