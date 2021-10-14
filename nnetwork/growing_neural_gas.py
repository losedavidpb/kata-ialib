import tensorflow as tf

class GrowingNeuralGas(object):

    def __init__(self, num_epochs=10):
        self.__num_epochs = num_epochs
        self.__A = None
        self.__N = []

    def fit(self, x_data):
        self.__A = tf.random.normal(shape=(2, x_data.shape[1]), mean=0.0, stdev=1.0, dtype=tf.float32)

        for epoch in tf.range(self.__num_epochs):
            pass

class Graph(object):

    def __init__(self, identifier: int, neighborhood=[], age_neighborhood=[]):
        self.__id = identifier
        self.__neighborhood = neighborhood
        self.__age_neighborhood = age_neighborhood

    @property
    def id(self): return self.__id

    @property
    def neighborhood(self): return self.__neighborhood

    @property
    def age_neighborhood(self): return self.__age_neighborhood

    def add_neighborhood(self, neighborhood, age_neighborhood):
        self.__neighborhood.append(neighborhood)
        self.__age_neighborhood.append(age_neighborhood)

    def remove_neighborhood(self, neighborhood):
        if neighborhood in self.__neighborhood:
            self.__age_neighborhood.pop(self.__neighborhood.index(neighborhood))
            self.__neighborhood.remove(neighborhood)
