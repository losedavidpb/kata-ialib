import numpy as np
import nnetwork.growing_neural_gas.growing_neural_gas as gng

# region _________________ GNG Loader _________________
class GrowingNeuralGasLoader(object):

    @staticmethod
    def load_gng(filename):
        file = np.load(filename, allow_pickle=True)
        model = gng.GrowingNeuralGas(
            epsilon_a=float(file['epsilon_a']), epsilon_n=float(file['epsilon_n']),
            a_max=int(file['a_max']), eta=float(file['eta']), alpha=float(file['alpha']),
            delta=float(file['delta']), max_number_units=int(file['max_number_units']),
            verbose=bool(file['verbose']), max_clusters=int(file['max_clusters']),
            number_epochs=int(file['number_epochs']))

        model.history = {'a': [], 'n': [], 'error': [], 'clusters': []}
        model.history['a'].extend(file['a'])
        model.history['n'].extend(file['n'])
        model.history['error'].extend(file['error'])
        model.history['clusters'].extend(file['clusters'])

        return model
# endregion

# region _________________ GNG Saver _________________
class GrowingNeuralGasSaver(object):

    @staticmethod
    def save_gng(filename, model):
        np.savez(filename, a=np.array(model.history['a']), n=np.array(model.history['n']),
                 error=np.array(model.history['error']), clusters=model.history['clusters'],
                 epsilon_a=model.epsilon_a, epsilon_n=model.epsilon_n, a_max=model.a_max,
                 eta=model.eta, alpha=model.alpha, number_epochs=model.number_epochs,
                 delta=model.delta, max_number_units=model.max_number_units,
                 max_clusters=model.max_clusters, verbose=model.verbose)
# endregion
