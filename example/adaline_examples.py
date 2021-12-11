from sklearn import datasets
from nnetwork.neurons import adaline
from visualizer import TwoDimensionDataPlotter
from visualizer import ErrorPlotter
from visualizer import NeuronPlotter
from visualizer import NeuronAnimation
import numpy as np

# region _________________ Global variables _________________
p_x_1 = np.array([
    [0.023, 0.112], [0.255, 0.111], [0.312, 0.234],
    [0.486, 0.533], [0.243, 0.335], [0.722, 0.545],
    [0.457, 0.223], [0.567, 0.678], [0.345, 0.789],
    [0.899, 0.666], [0.954, 0.444], [0.234, 0.777],
    [0.478, 0.854], [0.432, 0.555], [0.678, 0.977],
    [0.324, 0.555], [0.211, 0.246], [0.243, 0.131],
    [1.222, 1.222], [1.000, 0.777],

    [2.234, 2.232], [2.555, 2.222], [2.666, 2.111],
    [2.444, 2.777], [2.111, 2.111], [2.775, 2.345],
    [2.657, 2.212], [2.678, 2.346], [2.789, 2.346],
    [2.567, 2.555], [2.899, 2.211], [2.678, 2.678],
    [2.345, 2.124], [2.663, 2.435], [2.997, 2.999],
    [2.111, 2.345], [2.355, 2.467], [2.665, 2.352],
    [2.765, 2.113], [2.547, 2.145]
])

p_y_1 = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
])
# endregion

def test_1_adaline():
    print(">> Test Adaline: sample")
    TwoDimensionDataPlotter().init(x=p_x_1, y=p_y_1).show()

    # Adaline neuron works better for the classification of two classes,
    # since it was designed as a binary classifier
    model = adaline.AdalineGD(lr=0.001, n_epochs=2000)
    history = model.fit(p_x_1, p_y_1, max_tries=2)

    ErrorPlotter().init(errors=history['errors']).show()
    NeuronPlotter().init(x=p_x_1, y=p_y_1, weights=history['weights'][-1]).show()
    NeuronAnimation().init(x=p_x_1, y=p_y_1, weights=history['weights'], errors=history['errors']).show()

def test_2_adaline():
    print(">> Test Adaline: iris")
    iris = datasets.load_iris()
    x = iris.data[:, :2]
    y = iris.target

    # Load only two classes for classification
    x, y = x[y != 2], y[y != 2]

    TwoDimensionDataPlotter().init(x=x, y=y).show()

    model = adaline.AdalineGD(lr=0.0001, n_epochs=1000)
    history = model.fit(x, y, max_tries=2)

    ErrorPlotter().init(errors=history['errors']).show()
    NeuronPlotter().init(x=x, y=y, weights=history['weights'][-1]).show()
    NeuronAnimation().init(x=x, y=y, weights=history['weights'], errors=history['errors']).show()

if __name__ == '__main__':
    test_1_adaline()
    test_2_adaline()
