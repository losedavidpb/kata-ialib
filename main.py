from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

from nnetwork.neurons import adaline, perceptron, mcCulloch_pitts as mp

# _________________ Global variables _________________

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

# _________________ Visualization functions _________________

def _visualize_dataset(x, y):
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.title("Data visualization")
    plt.xlabel("x"), plt.ylabel("y")
    plt.grid(), plt.show()

def _visualize_model(x, y, model):
    weights = model.best_weights()

    print(x)
    print(model.predict(x, is_trained=True))

    # X values for line that represents the prediction
    start_p, stop_p = min(x[:, 0]), max(x[:, 0]) + 1

    line_data = (lambda m, n, _x: m * _x + n)(
        m=weights[1] / -weights[2],
        n=weights[0] / -weights[2],
        _x=np.arange(start=start_p, stop=stop_p)
    )

    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.plot(np.arange(start=start_p, stop=stop_p), line_data)
    plt.title("Optimal model visualization")
    plt.xlabel("x"), plt.ylabel("y")
    plt.grid(), plt.show()

def _visualize_errors(errors):
    plt.plot(errors, 'b-', label="Error SSE")
    plt.title("Error evolution")
    plt.xlabel("Iterations")
    plt.ylabel("Error (SSE)")
    plt.grid(), plt.show()

# ____________________ Tests for simple neurons  ____________________

def test_mcCulloch_pitts():
    p_x = np.array([[2.0, 3.0], [4.0, 1.0], [3.0, 1.0]])
    w = np.random.randint(low=-1, high=1, size=(p_x.shape[1]))
    w[w == 0] = 1

    model = mp.MPNeuron(theta=1)
    print("Prediction theta=1 => ", model.predict(weights=w, inputs=p_x))

def test_1_adaline():
    _visualize_dataset(p_x_1, p_y_1)

    # Adaline neuron works better for the classification of two classes,
    # since it was designed as a binary classifier
    model = adaline.AdalineGD(lr=0.001, n_epochs=2000)
    weights, errors = model.fit(p_x_1, p_y_1, max_tries=2)

    _visualize_errors(errors)
    _visualize_model(p_x_1, p_y_1, model)

def test_2_adaline():
    iris = datasets.load_iris()
    x = iris.data[:, :2]
    y = iris.target

    # Load only two classes for classification
    x, y = x[y != 2], y[y != 2]

    _visualize_dataset(x, y)

    model = adaline.AdalineGD(lr=0.0001, n_epochs=1000)
    _, errors = model.fit(x, y, max_tries=2)

    _visualize_errors(errors)
    _visualize_model(x, y, model)

def test_1_perceptron():
    _visualize_dataset(p_x_1, p_y_1)

    # Perceptron neuron works better for the classification of two classes,
    # since it was designed as a binary classifier
    model = perceptron.PerceptronGD(lr=0.01, n_epochs=30)
    weights, errors = model.fit(p_x_1, p_y_1, max_tries=2)

    _visualize_errors(errors)
    _visualize_model(p_x_1, p_y_1, model)

def test_2_perceptron():
    iris = datasets.load_iris()
    x = iris.data[:, :2]
    y = iris.target

    # Load only two classes for classification
    x, y = x[y != 2], y[y != 2]

    _visualize_dataset(x, y)

    model = perceptron.PerceptronGD(lr=0.001, n_epochs=100)
    _, errors = model.fit(x, y, max_tries=2)

    _visualize_errors(errors)
    _visualize_model(x, y, model)

if __name__ == '__main__':
    test_mcCulloch_pitts()
    test_1_adaline()
    test_2_adaline()
    test_1_perceptron()
    test_2_perceptron()
