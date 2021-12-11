from nnetwork.growing_neural_gas.growing_neural_gas import GrowingNeuralGas
from preprocessing import prepare_data
from sklearn import datasets
import tensorflow as tf

from visualizer import ErrorPlotter
from visualizer import GrowingNeuralGasPlotter
from visualizer import GrowingNeuralGasAnimation

def test_gng_1():
    print(">> Test GNG: iris")
    iris = datasets.load_iris()
    iris = iris.data[:, :]

    train_data, test_data = prepare_data(iris, remove_class=False)
    train_data = tf.Variable(train_data, dtype=tf.float32)
    test_data = tf.Variable(test_data, dtype=tf.float32)

    model = GrowingNeuralGas(
        epsilon_a=.1, epsilon_n=.05, a_max=5, eta=25, alpha=.1, delta=.1,
        max_number_units=100, max_clusters=10, number_epochs=100, verbose=True)

    history = model.fit(train_data)

    ErrorPlotter().init(errors=history['errors']).show()
    GrowingNeuralGasPlotter().init(history=history).show()
    GrowingNeuralGasAnimation().init(history=history).show()

    predicted_clusters = model.predict(test_data)

    for (xi, cluster), i in zip(predicted_clusters, range(0, len(test_data.shape))):
        print("* Row " + str(i) + ': ' + str(cluster))

def test_gng_2():
    print(">> Test GNG: breast_cancer")
    cancer = datasets.load_breast_cancer()
    cancer = cancer.data[:, :]

    train_data, test_data = prepare_data(cancer, remove_class=False)
    train_data = tf.Variable(train_data, dtype=tf.float32)
    test_data = tf.Variable(test_data, dtype=tf.float32)

    model = GrowingNeuralGas(
        epsilon_a=.1, epsilon_n=.05, a_max=5, eta=25, alpha=.1, delta=.1,
        max_number_units=100, max_clusters=10, number_epochs=100, verbose=True)

    history = model.fit(train_data)

    ErrorPlotter().init(errors=history['errors']).show()
    GrowingNeuralGasPlotter().init(history=history).show()
    GrowingNeuralGasAnimation().init(history=history).show()

    predicted_clusters = model.predict(test_data)

    for (xi, cluster), i in zip(predicted_clusters, range(0, len(test_data.shape))):
        print("* Row " + str(i) + ': ' + str(cluster))

if __name__ == '__main__':
    test_gng_1()
    test_gng_2()
