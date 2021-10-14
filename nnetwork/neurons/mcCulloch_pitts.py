import numpy as np

class MPNeuron(object):
    """
        McCulloch-Pitts neuron was the first computational model of a
        neuron proposed in 1943 by Warren McCulloch and Walter Pitts.
        It can be used as a logical unit and it is essential for the
        construction of a neuronal network.

        This neuron is a mathematical model inspired by biological
        neurons, so it has got connections to inputs (dendrites),
        a processing element that manipulates the inputs (soma),
        an activation function that makes a decision (axon), and
        finally connections to other neurons or the output (synapse).

        Talking about the model, the M-P Neuron can be divided into
        two sections: the first one is an aggregation of some elements
        passed to the neuron, and the second one takes the result of
        this operation and makes a decision (activation function),
        always depending on a value called "theta" which is the
        thresholding parameter.

        However, the limitations of this approximation explains why
        this computational model is not used nowadays.

            - Thresholding parameter needs to be hand coded
            - M-P Neuron is just useful for creating boolean functions
            - It is not possible to assign different importance to some inputs
            - Linearly separable functions can not be represented
            - There's no any feedback (no training process)
    """

    def __init__(self, theta):
        self.__theta = theta

    def predict(self, weights, inputs):
        net = np.sum(weights * inputs) - self.__theta
        return self.activation_f(net)

    def activation_f(self, u):
        return 1 if u >= self.__theta else 0
