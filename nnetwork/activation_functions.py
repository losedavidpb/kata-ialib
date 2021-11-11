import numpy as np

activation_function = {
    'lineal': (lambda x: x, lambda x: 1),
    'hard_sigmoid': (lambda x: np.max(0, np.min(1, (x + 1) / 2)), lambda x: 0 if x < -2.5 or x > 2.5 else 0.2),
    'sigmoid': (lambda x: 1 / (1 + np.exp(-x)), lambda x: (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))),
    'tanh': (lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)), lambda x: 1 - ((np.exp(x) - np.exp(-x))**2 / (np.exp(x) + np.exp(-x))**2)),
    'hard_tanh': (lambda x: -1 if x < -1 else (x if -1 <= x <= 1 else 1), lambda x: 1 if -1 <= x <= 1 else 0),
    'ReLU': (lambda x: (x > 0) * x, lambda x: (x > 0) * 1),
    'leaky_ReLU': (lambda x: 0.01 * x if x < 0 else x, lambda x: 0.01 if x < 0 else 1),
    'pReLU': (lambda x, alpha: alpha * x if x <= 0 else x, lambda x, alpha: alpha if x <= 0 else 1),
    'rRelu': (lambda x: x if x >= 0 else np.random.uniform(0.8, 0.3) * x, lambda x: 1 if x >= 0 else np.random.uniform(0.8, 0.3) * 1),
    'elu': (lambda x: x if x > 0 else 0.01 * (np.exp(x) - 1), lambda x: 1 if x > 0 else 0.01 * np.exp(x)),
    'swish': (lambda x: x * (1 / (1 + np.exp(-x))), None),
    'softsign': (lambda x: x / (1 + np.abs(x)), lambda x: 1 / (1 + np.abs(x))**2),
    'softplus': (lambda x: np.log(1 + np.exp(x)), lambda x: 1 / (1 + np.exp(-x)))
}

def get_activation(name_function):
    if name_function in activation_function.keys():
        activation_f, _ = activation_function[name_function]
        return activation_f

def get_derivative_activation(name_function):
    if name_function in activation_function.keys():
        _, derivative_activation_f = activation_function[name_function]
        return derivative_activation_f
