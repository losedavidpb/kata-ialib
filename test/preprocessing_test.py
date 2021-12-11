import unittest
import numpy as np
from preprocessing import split_data, principal_components

class PreprocessingTest(unittest.TestCase):

    def test_split_data_when_no_shuffle(self):
        test_data = np.random.random(size=(10, 10))
        base_train_data = test_data[0:9, :]
        base_test_data = test_data[9:10, :]

        test_train_data, test_test_data =\
            split_data(test_data, train_size=.90, shuffle=False)

        self.assertEqual(base_train_data.tolist(), test_train_data.tolist())
        self.assertEqual(base_test_data.tolist(), test_test_data.tolist())

    def test_split_data_when_shuffle(self):
        test_data = np.random.random(size=(10, 10))
        num_train_data = 9 * test_data.shape[1]
        num_test_data = 1 * test_data.shape[1]

        test_train_data, test_test_data =\
            split_data(test_data, train_size=.90, shuffle=True)

        self.assertEqual(num_train_data, test_train_data.size)
        self.assertEqual(num_test_data, test_test_data.size)

    def test_principal_components(self):
        data = np.random.random(size=(20, 20))

        for i in range(0, np.random.randint(low=2, high=10)):
            rand_row = np.random.randint(low=0, high=9)
            rand_col = np.random.randint(low=0, high=9)
            data[rand_row, rand_col] = None

        num_duplicates = 10

        for i in range(0, data.shape[0]):
            if num_duplicates == data.shape[0]: break

            if not np.all(np.isnan(data[i, :])):
                data[num_duplicates, :] = data[num_duplicates - int(data.shape[0] / 2), :]
                num_duplicates = num_duplicates + 1

        data_pca = principal_components(data, remove_class=False)
        self.assertFalse(np.all(np.isnan(data_pca[:, :])))
        self.assertTrue(data_pca.shape[0] <= int(data.shape[0] / 2))

if __name__ == '__main__':
    unittest.main()
