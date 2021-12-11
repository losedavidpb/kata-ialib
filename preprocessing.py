from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

def split_data(data, train_size=0.95, shuffle=True):
    num_train_data = int(train_size * data.shape[0])
    if shuffle is True: np.random.shuffle(data)

    train_data = data[0:num_train_data, :]
    test_data = data[num_train_data:data.shape[0], :]
    return train_data, test_data

def principal_components(data, remove_class=True):
    _data = pd.DataFrame(data).drop_duplicates().dropna()
    if remove_class: _data = _data.drop('Class', axis=1)
    _data = np.array(_data.iloc[:, :], dtype=float)

    _data = StandardScaler().fit_transform(_data)
    _data = PCA(n_components=None).fit_transform(_data)
    return _data

def prepare_data(data, remove_class=True, train_size=.95, shuffle=True):
    data = principal_components(data, remove_class)
    return split_data(data, train_size, shuffle)
