import time
import numpy as np
import tensorflow as tf
from nnetwork.growing_neural_gas.gng_graph import GraphGNG
from utils import get_decimal_format


class GrowingNeuralGas(object):

    def __init__(self, epsilon_a=.1, epsilon_n=.05, a_max=25, eta=25, alpha=.1, delta=.1,
                 max_number_units=10, max_clusters=2, number_epochs=100, verbose=True):

        self.epsilon_a, self.epsilon_n = epsilon_a, epsilon_n
        self.a_max, self.eta = a_max, eta
        self.alpha, self.delta = alpha, delta
        self.max_number_units = max_number_units
        self.max_clusters = max_clusters
        self.number_epochs = number_epochs
        self.verbose = verbose
        self.history = {}

    @staticmethod
    def increment_age_neighborhood(index_nearest_unit, n):
        n[index_nearest_unit].increment_age_neighborhood(1.0)

        for index_neighbor in n[index_nearest_unit].neighborhood:
            n[index_neighbor].increment_age_neighbor(index_nearest_unit, 1.0)

        return n

    @staticmethod
    def find_nearest_unit(xi, a):
        return (tf.math.argmin(tf.math.reduce_sum(tf.math.pow(a - xi, 2), 1))).numpy()

    @staticmethod
    def find_second_nearest_unit(xi, a):
        index_nearest_unit = GrowingNeuralGas.find_nearest_unit(xi, a)
        error_ = tf.constant(tf.math.reduce_sum(tf.math.pow(a - xi, 2), 1), dtype=tf.float32).numpy()
        error_[index_nearest_unit] = np.Inf
        return (tf.math.argmin(tf.constant(error_))).numpy()

    @staticmethod
    def find_index_neighbor_max_error(index_unit_with_max_error_, n, error_):
        index = (tf.squeeze(tf.math.argmax(tf.gather(error_, n[index_unit_with_max_error_].neighborhood)), 0)).numpy()
        index_neighbour_max_error = n[index_unit_with_max_error_].neighborhood[index]
        return index_neighbour_max_error

    @staticmethod
    def prune_a(a, n):
        index_to_not_remove, new_n = [index for index in tf.range(len(n)) if len(n[index].neighborhood) > 0], []
        a = tf.Variable(tf.gather(a, index_to_not_remove, axis=0))

        for index in reversed(range(n.__len__())):
            if n[index].neighborhood.__len__() == 0:
                for pivot in range(index + 1, n.__len__()):
                    n[pivot].identifier = n[pivot].identifier - 1

                    for index_n in range(n.__len__()):
                        for index_neighborhood in range(n[index_n].neighborhood.__len__()):
                            if n[index_n].neighborhood[index_neighborhood] == pivot:
                                n[index_n].neighborhood[index_neighborhood] -= 1

                n.pop(index)

        return a, n

    @staticmethod
    def _get_cluster_from(source, n, clusters):
        cluster, closed, stack = [], {}, [source.identifier]

        while len(stack) != 0:
            identifier_graph = stack.pop(0)
            closed.setdefault(identifier_graph, False)

            if identifier_graph not in cluster:
                check_id = [identifier_graph in clusters[c] for c in clusters.keys()]
                if any(iter(check_id)): return []

                cluster.append(identifier_graph)
                closed[identifier_graph] = True

            for neighbor in n[identifier_graph].neighborhood:
                closed.setdefault(neighbor, False)

                if closed[neighbor] is not True:
                    stack.append(neighbor)

        return cluster

    @staticmethod
    def _get_clusters(n):
        clusters, id_cluster = {}, 0

        for i in tf.range(0, len(n)):
            cluster = GrowingNeuralGas._get_cluster_from(n[i], n, clusters)

            if cluster.__len__() != 0:
                clusters[id_cluster] = cluster
                id_cluster = id_cluster + 1

        total_elements = np.sum([len(clusters[c]) for c in clusters.keys()])
        assert np.abs(total_elements - len(n)) == 0

        return clusters

    @staticmethod
    def total_error(error):
        return tf.Variable(tf.reduce_mean(error)).numpy()

    def predict(self, test_x):
        _test_x = tf.Variable(test_x, dtype=tf.float32)

        n, a = self.history['n'][-1], self.history['a'][-1]
        clusters = self.history['clusters'][-1]
        predicted_clusters = []

        for i in tf.range(0, _test_x.shape[0]):
            xi = _test_x[i]

            nearest_index = GrowingNeuralGas.find_nearest_unit(xi, a)
            cluster_arr = [c for c in range(0, len(clusters.keys())) if n[nearest_index].identifier in clusters[c]]
            assert len(cluster_arr) == 1

            cluster = cluster_arr[0] if len(cluster_arr) == 1 else None
            predicted_clusters.append((test_x, cluster))

        return predicted_clusters

    def fit(self, training_x):
        _training_x = tf.Variable(training_x, dtype=tf.float32)

        str_format = "Epoch {}: num_clusters={}, number_units={}, error={}"

        self.history.clear()
        self.history = {'a': [], 'n': [], 'errors': [], 'clusters': []}
        self.history['a'].append(tf.Variable(tf.random.normal([2, _training_x.shape[1]], 0.0, 1.0, dtype=tf.float32)))
        self.history['n'].append([GraphGNG(0), GraphGNG(1)])
        self.history['errors'].append(tf.Variable(tf.zeros([2, 1]), dtype=tf.float32))
        epoch, number_processed_row, num_clusters = 0, 0, 1

        while epoch < self.number_epochs and num_clusters < self.max_clusters:
            if self.history['a'][epoch].shape[0] > self.max_number_units: break

            a, n = tf.Variable(self.history['a'][epoch], dtype=tf.float32), []
            error_ = tf.Variable(self.history['errors'][epoch])

            for graph in self.history['n'][epoch]:
                n.append(graph.__copy__())

            shuffled_training_x = tf.random.shuffle(_training_x)

            for row_ in tf.range(shuffled_training_x.shape[0]):
                if self.verbose is True:
                    porc_value = round((((row_ + 1) * 100) / shuffled_training_x.shape[0]).numpy(), 2)
                    print('\r\x1b[2K' + "Epoch " + str(epoch) + ": " + str(porc_value) + "%", end='')
                    time.sleep(0.001)

                xi = shuffled_training_x[row_]

                index_nearest_unit = self.find_nearest_unit(xi, a.value())
                index_second_nearest_unit = self.find_second_nearest_unit(xi, a.value())
                n = self.increment_age_neighborhood(index_nearest_unit, n)

                error_[index_nearest_unit].assign(
                    error_[index_nearest_unit] + tf.math.reduce_sum(
                        tf.math.squared_difference(xi, a[index_nearest_unit])
                    )
                )

                a[index_nearest_unit].assign(
                    a[index_nearest_unit] + self.epsilon_a * (xi - a[index_nearest_unit])
                )

                for index_neighbour in n[index_nearest_unit].neighborhood:
                    a[index_neighbour].assign(
                        a[index_neighbour] + self.epsilon_n * (xi - a[index_neighbour])
                    )

                if index_second_nearest_unit in n[index_nearest_unit].neighborhood:
                    n[index_nearest_unit].set_age(index_second_nearest_unit, 0.0)
                    n[index_second_nearest_unit].set_age(index_nearest_unit, 0.0)
                else:
                    n[index_nearest_unit].add_neighbour(index_second_nearest_unit, 0.0)
                    n[index_second_nearest_unit].add_neighbour(index_nearest_unit, 0.0)

                for graph in n:
                    graph.prune_graph(self.a_max)

                a, n = self.prune_a(a, n)

                if not (number_processed_row + 1) % self.eta:
                    index_unit_with_max_error_ = tf.squeeze(tf.math.argmax(error_), 0).numpy()
                    index_neighbour_with_max_error_ = self.find_index_neighbor_max_error(
                        index_unit_with_max_error_, n, error_)

                    a = tf.Variable(tf.concat([a, tf.expand_dims(
                        0.5 * (a[index_unit_with_max_error_] + a[index_neighbour_with_max_error_]), 0)], 0))

                    max_error = [index_unit_with_max_error_, index_neighbour_with_max_error_]
                    n.append(GraphGNG(int(a.shape[0] - 1), max_error, [0.0, 0.0]))

                    n[index_unit_with_max_error_].remove_neighbour(index_neighbour_with_max_error_)
                    n[index_unit_with_max_error_].add_neighbour(a.shape[0] - 1, 0.0)

                    n[index_neighbour_with_max_error_].remove_neighbour(index_unit_with_max_error_)
                    n[index_neighbour_with_max_error_].add_neighbour(a.shape[0] - 1, 0.0)

                    error_[index_unit_with_max_error_].assign(error_[index_unit_with_max_error_] * self.alpha)
                    error_[index_neighbour_with_max_error_].assign(error_[index_neighbour_with_max_error_] * self.alpha)
                    error_ = tf.Variable(tf.concat([error_,  tf.expand_dims(error_[index_unit_with_max_error_], 0)], 0))

                error_.assign(error_.value() * self.delta)
                number_processed_row += 1

            clusters = self._get_clusters(n)
            num_clusters, error_mean = len(clusters.keys()), self.total_error(error_)

            if self.verbose is True:
                print('\r\x1b[2K' + "Epoch " + str(epoch) + ": 100%", end='')
                time.sleep(0.001)

                error_mean = get_decimal_format(error_mean)
                print('\r\x1b[2K' + str.format(str_format, epoch, num_clusters, a.shape[0], error_mean))

            self.history['a'][epoch] = tf.Variable(a)
            self.history['n'][epoch] = [graph.__copy__() for graph in n]
            self.history['errors'][epoch] = tf.Variable(error_)

            self.history['a'].append(a)
            self.history['n'].append(n)
            self.history['errors'].append(error_)
            self.history['clusters'].append(clusters)
            epoch = epoch + 1

        self.history['errors'] = [
            self.total_error(self.history['errors'][i])
            for i in tf.range(0, len(self.history['errors']))
        ]

        return self.history
