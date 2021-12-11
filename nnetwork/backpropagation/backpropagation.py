from nnetwork.activation_functions import get_activation, get_derivative_activation
from nnetwork.backpropagation import input_layer
from nnetwork.backpropagation import hidden_layer
from nnetwork.backpropagation import output_layer
import numpy as np
import tensorflow as tf

class BackPropagation(object):

    def __init__(self, input_activation, hidden_activation, output_activation, eta=.01, num_epochs=50, num_batch=50):
        self.eta, self.num_epochs, self.num_batch = eta, num_epochs, num_batch
        self.input_activation = self._init_activation_f(input_activation)
        self.hidden_activation = self._init_activation_f(hidden_activation)
        self.output_activation = self._init_activation_f(output_activation)
        self.input_layer, self.hidden_layers, self.output_layer = None, None, None
        self.history = {}

    @staticmethod
    def _init_activation_f(activation_name):
        activation_f = get_activation(activation_name)
        derivative_activation_f = get_derivative_activation(activation_name)
        return [activation_f, derivative_activation_f]

    @staticmethod
    def _get_output_delta(_output_layer, p_Y):
        return np.subtract(p_Y, _output_layer.last_output) * _output_layer.derivative_activation()

    @staticmethod
    def _get_delta(last_delta, last_weights):
        return np.dot(last_delta, last_weights.T[:, 1:])

    @staticmethod
    def _get_diff_weights(previous_layer, eta, delta):
        diff_weights_rest = eta * np.dot(delta.T, previous_layer.last_output)
        diff_weights_0 = eta * delta.T
        return np.hstack([diff_weights_0, diff_weights_rest])

    def _init_layers(self, train_data, num_hidden_layers, num_neurons_hidden_layers):
        self.input_layer = input_layer.InputLayer(
            activation=self.input_activation[0],
            derivative_activation=self.input_activation[1])

        self.input_layer.init_weights()
        self.hidden_layers = []

        for v_layer in range(0, num_hidden_layers):
            _num_inputs_each_neuron = self.input_layer.num_neurons
            if v_layer != 0: _num_inputs_each_neuron = num_neurons_hidden_layers[v_layer - 1]

            self.hidden_layers.append(
                hidden_layer.HiddenLayer(
                    activation=self.hidden_activation[0],
                    derivative_activation=self.hidden_activation[1],
                    num_neurons=num_neurons_hidden_layers[v_layer],
                    num_inputs_each_neuron=_num_inputs_each_neuron
                )
            )

            self.hidden_layers[v_layer].init_weights()

        self.output_layer = output_layer.OutputLayer(
            activation=self.output_activation[0],
            derivative_activation=self.output_activation[1],
            num_neurons=train_data.shape[1],
            num_inputs_each_neuron=self.hidden_layers[num_hidden_layers - 1].num_neurons)

        self.output_layer.init_weights()

    def _forward_propagate(self, p_x):
        return self.predict(p_x)

    def _backward_propagate(self, p_y):
        output_delta = self._get_output_delta(self.output_layer, p_y)
        self.output_layer.weights += (self._get_diff_weights(self.hidden_layers[-1], self.eta, output_delta)).T
        last_weights, last_delta = self.output_layer.weights, output_delta

        for i in reversed(range(0, len(self.hidden_layers))):
            previous_layer = self.input_layer if i == 0 else self.hidden_layers[i - 1]

            delta = self._get_delta(last_delta, last_weights)
            self.hidden_layers[i].weights += (self._get_diff_weights(previous_layer, self.eta, delta)).T

            last_weights, last_delta = self.hidden_layers[i].weights, delta

    @staticmethod
    def total_accuracy(predicted, test):
        _predicted = np.array(predicted.numpy() if tf.is_tensor(predicted) else predicted)
        n_hits = len([1 for predicted, expected in zip(predicted, test) if predicted == expected])
        return np.round(n_hits * 100 / len(test), 2)

    @staticmethod
    def total_error(predicted, last_output):
        _predicted = np.array(predicted.numpy() if tf.is_tensor(predicted) else predicted)
        _last_output = np.array(last_output.numpy() if tf.is_tensor(last_output) else last_output)
        return np.mean(np.square(np.subtract(_predicted, _last_output)))

    def predict(self, p_x):
        _p_x = np.array(p_x.numpy() if tf.is_tensor(p_x) else p_x)
        v_Y_input_layer_ = self.input_layer.predict(_p_x)
        v_X_hidden_layer_ = v_Y_input_layer_
        v_Y_hidden_layer_ = None

        for v_hidden_layer in self.hidden_layers:
            v_Y_hidden_layer_ = v_hidden_layer.predict(v_X_hidden_layer_)
            v_X_hidden_layer_ = v_Y_hidden_layer_

        v_X_output_layer_ = v_Y_hidden_layer_
        v_Y_output_layer_ = self.output_layer.predict(v_X_output_layer_)
        return v_Y_output_layer_

    def fit(self, x_train, y_train, x_test, y_test, num_hidden_layers, num_neurons_hidden_layers=np.array([1])):
        _x_train = np.array(x_train.numpy() if tf.is_tensor(x_train) else x_train)
        _y_train = np.array(y_train.numpy() if tf.is_tensor(y_train) else y_train)
        _x_test = np.array(x_test.numpy() if tf.is_tensor(x_test) else x_test)
        _y_test = np.array(y_test.numpy() if tf.is_tensor(y_test) else y_test)
        str_format = "Epoch {}: accuracy={}%, loss={}%"

        self.history.clear()
        self.history = {'accuracy': [], 'error': []}

        self._init_layers(_y_train, num_hidden_layers, num_neurons_hidden_layers)

        for epoch in range(0, self.num_epochs):
            rand_pos = np.random.permutation(len(_x_train))
            x_train_i, y_train_i = _x_train[rand_pos], _y_train[rand_pos]

            for batch in range(0, self.num_batch):
                x_train_b, y_train_b = x_train_i[batch, :], y_train_i[batch, :]
                self._forward_propagate(np.resize(a=x_train_b, new_shape=(1, len(x_train_b))))
                self._backward_propagate(np.resize(a=y_train_b, new_shape=(1, len(y_train_b))))

            prediction_x = self.predict(_x_test)
            self.history['error'].append(self.total_error(_y_test, self.output_layer.last_output))
            self.history['accuracy'].append(self.total_accuracy(prediction_x, _y_test))

            accuracy_epoch = np.round(self.history['accuracy'][epoch], 4)
            error_epoch = np.round(self.history['error'][epoch], 4)
            print(str.format(str_format, epoch, accuracy_epoch, error_epoch))

        return self
