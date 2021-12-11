import unittest
from nnetwork.growing_neural_gas.gng_graph import GraphGNG

class GraphTest(unittest.TestCase):

    def test_add_neighborhood(self):
        graph_base = GraphGNG(7, [1, 2, 3], [10, 20, 30])
        graph_test = GraphGNG(7)
        graph_test.add_neighbour(1, 10)
        graph_test.add_neighbour(2, 20)
        graph_test.add_neighbour(3, 30)

        self.assertEqual(graph_base, graph_test)

    def test_remove_neighborhood(self):
        graph_base = GraphGNG(7, [1, 3], [10, 30])
        graph_test = GraphGNG(7, [1, 2, 3], [10, 20, 30])
        graph_test.remove_neighbour(2)

        self.assertEqual(graph_base, graph_test)

    def test_increment_age_neighborhood(self):
        graph_base = GraphGNG(7, [1, 2, 3], [11, 21, 31])
        graph_test = GraphGNG(7, [1, 2, 3], [10, 20, 30])
        graph_test.increment_age_neighborhood(1)

        self.assertEqual(graph_base, graph_test)

    def test_increment_age_neighbor(self):
        graph_base = GraphGNG(7, [1, 2, 3], [10.0, 21.0, 30.0])
        graph_test = GraphGNG(7, [1, 2, 3], [10.0, 20.0, 30.0])
        graph_test.increment_age_neighbor(2, 1.0)

        self.assertEqual(graph_base, graph_test)

    def test_increment_age_neighborhood_when_graph_have_not_neighborhood(self):
        graph_base = GraphGNG(7, [], [])
        graph_test = GraphGNG(7, [], [])
        graph_test.increment_age_neighborhood(1)

        self.assertEqual(graph_base, graph_test)

    def test_set_age_neighborhood(self):
        graph_base = GraphGNG(7, [1, 2, 3], [11, 21, 31])
        graph_test = GraphGNG(7, [1, 2, 3], [22, 10, 3])
        graph_base.set_age(1, 22)
        graph_base.set_age(2, 10)
        graph_base.set_age(3, 3)

        self.assertEqual(graph_base, graph_test)

    def test_prune_graph(self):
        graph_base_30 = GraphGNG(7, [1, 2, 4, 5], [25, 20, 25, 20])
        graph_base_25 = GraphGNG(7, [2, 5], [20, 20])
        graph_bae_20 = GraphGNG(7)

        graph_test = GraphGNG(7, [1, 2, 3, 4, 5], [25.0, 20.0, 30.0, 25.0, 20.0])

        graph_test.prune_graph(30)
        self.assertEqual(graph_test, graph_base_30)

        graph_test.prune_graph(25)
        self.assertEqual(graph_test, graph_base_25)

        graph_test.prune_graph(20)
        self.assertEqual(graph_test, graph_bae_20)

if __name__ == '__main__':
    unittest.main()
