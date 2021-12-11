import unittest
import tensorflow as tf
from nnetwork.growing_neural_gas.growing_neural_gas import GrowingNeuralGas
from nnetwork.growing_neural_gas.gng_graph import GraphGNG

class GrowingNeuralGasTest(unittest.TestCase):

    def test_find_nearest_unit_and_find_second_nearest_unit(self):
        a = tf.constant([[0., 0., 0.], [0., 1., 0.], [0.5, 0.1, 0.], [2., 3., 0.5], [0.1, 0.5, 0.5]], dtype=tf.float32)
        xi = tf.expand_dims(tf.constant([0.45, 0.15, 0.05]), 0)

        growing_neural_gas = GrowingNeuralGas()
        index_nearest_unit = growing_neural_gas.find_nearest_unit(xi, a)
        index_second_nearest_unit = growing_neural_gas.find_second_nearest_unit(xi, a)

        self.assertEqual(index_nearest_unit, 2)
        self.assertEqual(index_second_nearest_unit, 0)

    def test_pruneA(self):
        a_base = tf.constant([[0., 0., 0.], [0.5, 0.1, 0.], [0.1, 0.5, 0.5]], dtype=tf.float32)
        n_base = [GraphGNG(0, [1, 2], [71, 31]), GraphGNG(1, [0, 2], [21, 41]), GraphGNG(2, [0, 1], [11, 32])]

        a_test = tf.Variable([[0., 0., 0.], [0., 1., 0.], [0.5, 0.1, 0.], [2., 3., 0.5], [0.1, 0.5, 0.5]], dtype=tf.float32)
        n_test = [GraphGNG(0, [2, 4], [71, 31]), GraphGNG(1), GraphGNG(2, [0, 4], [21, 41]), GraphGNG(3), GraphGNG(4, [0, 2], [11, 32])]

        growing_neural_gas = GrowingNeuralGas()
        a, n = growing_neural_gas.prune_a(a_test, n_test)

        self.assertTrue(tf.math.reduce_all(a_base == tf.constant(a)))
        for graphBase, graphTest in zip(n_base, n):
            self.assertEqual(graphBase, graphTest)

if __name__ == '__main__':
    unittest.main()
