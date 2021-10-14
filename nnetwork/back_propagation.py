import numpy as np
import abc

# _____________________ BackPropagation with Gradient Descent _____________________

class BackPropagationGD(object):

    def __init__(self, layer_types, lr=0.01, n_epochs=10):
        self.__input_layer, self.__h_layers, self.__output_layer = None, None, None
        self.__layer_types = layer_types
        self.__errors = []
        self.__n_epochs = n_epochs
        self.__lr = lr

    def _init_layers(self, p_x, p_y, n_h_layers=1, n_neurons_h_layers=np.array([1])):
        input_layer, h_layer, output_layer = self.__layer_types

        # Creating all layers for current neuronal network
        self.__input_layer = input_layer(n_neurons=p_x.shape[1])
        self.__h_layers = []

        for i in range(0, n_h_layers):
            n_neurons_i = n_neurons_h_layers[i]
            n_inputs = p_x.shape[1] if i == 0 else n_neurons_h_layers[i - 1]
            self.__h_layers.append(h_layer(n_neurons_i, n_inputs))

        self.__output_layer = output_layer(p_y.shape[1], n_neurons_h_layers[n_h_layers - 1])

        # Init weights for each layer of current neuronal network
        n_size = np.random.RandomState(None)
        self.__input_layer.init_weights(n_size)

        for h_layer in self.__h_layers:
            h_layer.init_weights(n_size)

        self.__output_layer.init_weights(n_size)

    def _forward_propagate(self, p_x):
        pass

    def _backward_propagate_error(self):
        pass

    def _update_weights(self):
        pass

    @abc.abstractmethod
    def fit(self, x_train, y_train, x_test, y_test, n_h_layers=1, n_neurons_h_layers=np.array([1])):
        self._init_layers(x_train, y_train, n_h_layers, n_neurons_h_layers)
        self.__errors = []

        for _ in range(0, self.__n_epochs):
            self._forward_propagate(x_train)
            self.__errors.append(self._backward_propagate_error())
            self._update_weights()

        # Test neural network

        # ...

        return self.__errors

    def predict(self, p_x):
        vy_input_layer = self.__input_layer.predict(p_x)
        vx_hidden_layer = vy_input_layer
        vy_hidden_layer = None

        for v_hidden_layer in self.__h_layers:
            vy_hidden_layer = v_hidden_layer.predict(vx_hidden_layer)
            vx_hidden_layer = vy_hidden_layer

        vx_output_layer = vy_hidden_layer
        vy_output_layer = self.__output_layer.predict(vx_output_layer)
        return vy_output_layer
