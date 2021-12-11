from nnetwork.neurons import mcCulloch_pitts as mp
import numpy as np

def test_mcCulloch_pitts():
    p_x = np.array([[2.0, 3.0], [4.0, 1.0], [3.0, 1.0]])
    w = np.random.randint(low=-1, high=1, size=(p_x.shape[1]))
    w[w == 0] = 1

    model = mp.MPNeuron(theta=1)
    print("Prediction theta=1 => ", model.predict(weights=w, inputs=p_x))

if __name__ == '__main__':
    test_mcCulloch_pitts()
