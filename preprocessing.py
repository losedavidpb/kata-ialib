from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import tensorflow as tf

def split_data(data, train_size=0.95, shuffle=True):
    _data = np.array(data.numpy() if tf.is_tensor(data) else data)
    num_train_data = int(train_size * _data.shape[0])
    if shuffle is True: np.random.shuffle(_data)

    train_data = np.array(_data[0:num_train_data, :])
    test_data = np.array(_data[num_train_data:_data.shape[0], :])
    return train_data, test_data

def normalize_data(data):
    _data = np.array(data.numpy() if tf.is_tensor(data) else data)
    _data = StandardScaler().fit_transform(_data)
    return _data

def principal_components(data, remove_class=True):
    _data = np.array(data.numpy() if tf.is_tensor(data) else data)
    _data = pd.DataFrame(_data).drop_duplicates().dropna()
    if remove_class: _data = _data.drop('Class', axis=1)
    _data = np.array(_data.iloc[:, :], dtype=float)

    _data = normalize_data(_data)

    porc_info, num_components = 0, 2
    components = None

    while porc_info < 0.85 and num_components < _data.shape[1]:
        pca = PCA(n_components=num_components)
        components = pca.fit_transform(_data)
        expl = pca.explained_variance_ratio_
        porc_info = np.sum(expl[0:16])
        num_components = num_components + 1

    _data = np.array(components, dtype=float)
    return _data

def prepare_data(data, remove_class=True, train_size=.95, shuffle=True):
    pca_data = principal_components(data, remove_class)
    return split_data(pca_data, train_size, shuffle)
