import tensorflow as tf

class GrowingNeuralGas(object):

    def __init__(self, epsilon_b, epsilon_n, a_max, eta, alpha, delta, num_epochs=10):
        self.__epsilon_b, self.__epsilon_n = epsilon_b, epsilon_n
        self.__a_max, self.__eta = a_max, eta
        self.__alpha, self.__delta = alpha, delta
        self.__num_epochs = num_epochs
        self.__nodes, self.__edges = None, []
        self.__error = None

    def find_first_second_nearest_unit(self, param):
        return None, None

    def fit(self, x_data):
        # Initialize first two units with its graphs associated
        self.__nodes = tf.Variable(tf.random.normal(shape=(2, x_data.shape[1]), mean=0.0, stdev=1.0, dtype=tf.float32))
        self.__edges.append(Graph(0))
        self.__edges.append(Graph(1))
        self.__error = tf.Variable(tf.zeros(shape=(x_data.shape[1]), dtype=tf.float32))

        for _ in tf.range(self.__num_epochs):
            # Create an input dataset with a probability function
            shuffled_x_data = tf.random.shuffle(x_data)

            for row in tf.range(shuffled_x_data.shape[0]):
                shuffled_x_data_i = shuffled_x_data[row]

                # Find two nearest units s1 and s2 for this iteration
                index_s1, index_s2 = self.find_first_second_nearest_unit(shuffled_x_data_i)

                # Increment age weight for first nearest unit from the rest of units
                self.__edges[index_s1].increment_age_weight()

                # Add error value for s1 unit and input value generated
                self.__error.assign(self.__error + tf.math.reduce_sum(tf.math.squared_difference(shuffled_x_data_i, shuffled_x_data[index_s1])))

                # Move s1 unit and all its neighborhoods to input generated using epsilon constants
                self.__nodes[index_s1].assign(self.__nodes[index_s1] + self.__epsilon_b * (shuffled_x_data_i - shuffled_x_data[index_s1]))
                self.__nodes[self.__edges[index_s1].neighborhood].assign(self.__edges[index_s1].neighborhood + self.__epsilon_n * (shuffled_x_data_i - shuffled_x_data[self.__edges[index_s1].neighborhood]))


# region Graph Implementation for GNG
class Graph(object):

    def __init__(self, identifier: int, neighborhood=[], age_neighborhood=[]):
        self.__id = identifier
        self.__neighborhood = neighborhood
        self.__age_neighborhood = age_neighborhood

    @property
    def id(self):
        return self.__id

    @property
    def neighborhood(self):
        return self.__neighborhood

    @property
    def age_neighborhood(self): return self.__age_neighborhood

    def increment_age_weight(self):
        pass

    def add_neighborhood(self, neighborhood, age_neighborhood):
        self.__neighborhood.append(neighborhood)
        self.__age_neighborhood.append(age_neighborhood)

    def remove_neighborhood(self, neighborhood):
        if neighborhood in self.__neighborhood:
            self.__age_neighborhood.pop(self.__neighborhood.index(neighborhood))
            self.__neighborhood.remove(neighborhood)
# endregion
