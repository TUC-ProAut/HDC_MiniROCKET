# Kenny Schlegel, Peer Neubert, Peter Protzel
#
# HDC-MiniROCKET : Explicit Time Encoding in Time Series Classification with Hyperdimensional Computing
# Copyright (C) 2022 Chair of Automation Technology / TU Chemnitz

import numpy as np
from data.constants import *
import pandas as pd
from sktime.datasets import load_UCR_UEA_dataset
from sklearn.model_selection import train_test_split
from models.HDC_MINIROCKET import HDC_MINIROCKET_model
from models.Minirocket_utils.minirocket_multivariate import MiniRocketMultivariate
from config import *
from sklearn.metrics.pairwise import cosine_similarity

####################################
# datasets loading
####################################
def load_dataset(dataset,config):
    """
    load the specific data set (from the data/ folder)
    @param dataset: specifies the data set [string]
    @param config: configure struct with necessary parameters [struct]
    @return: set of training and test data [tuple]
    """
    # load preprocessed data
    if dataset == "UCR":
        X_train, X_test, y_train, y_test = load_UCR_dataset(ucr_index=config.ucr_idx)
    elif dataset == "synthetic":
        X_train, X_test, y_train, y_test = load_synthetic_dataset(hard_case=False)
    elif dataset == "synthetic_hard":
        X_train, X_test, y_train, y_test = load_synthetic_dataset(hard_case=True)
    else:
        print("No valid dataset argument was set!")

    # remove nan if in dataset
    X_train[np.isnan(X_train)] = 0
    X_test[np.isnan(X_test)] = 0

    if config.normalize_data:
        print('Normalize data...')
        # Normalizing dimensions independently
        for j in range(X_train.shape[1]):
            mean = np.mean(X_train[:, j])
            std = np.std(X_train[:, j])
            std = np.where(std ==0 , 1, std)
            X_train[:, j] = (X_train[:, j] - mean) / std
            X_test[:, j] = (X_test[:, j] - mean) / std

    return X_train, X_test, y_train, y_test, config


def load_UCR_dataset(ucr_index):
    """
    """
    print("Loading UCR train / test dataset : ", UCR_SETS[ucr_index])

    X_train, y_train = load_UCR_UEA_dataset(name=UCR_PREFIX[ucr_index], split='train',return_X_y=True,extract_path=data_path_univar)
    X_test, y_test = load_UCR_UEA_dataset(name=UCR_PREFIX[ucr_index], split='test',return_X_y=True,extract_path=data_path_univar)

    X_train, X_test = df2np(X_train, X_test)

    # concat label to create different classes
    labels = np.concatenate((y_train,y_test))
    labels = pd.Categorical(pd.factorize(labels)[0])
    y_train = np.array(labels[0:y_train.shape[0]])
    y_test = np.array(labels[y_train.shape[0]:])

    print("Finished processing train dataset..")

    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    nb_dims = X_train.shape[1]
    length = X_train.shape[2]

    print()
    print("Number of train samples : ", train_size, "Number of test samples : ", test_size)
    print("Number of variables : ", nb_dims)
    print("Sequence length : ", length)

    return X_train, X_test, y_train, y_test

def load_synthetic_dataset(hard_case=False):
    '''
    create synthetic dataset with impulse as signal and additional noise
    @param hard_case: if set to true, the output is a difficult dataset selected from the original synthetic samples
    @return: tuple of training and test samples and the labels (X_train, X_test, y_train, y_test)
    '''
    ## create synthetic signal
    length = 500
    a = 0.03

    t = np.linspace(-10, 10, length)
    X_ = np.zeros((length, 1, length), dtype=np.float32)

    for i in range(length):
        X_[i, :, :] = np.roll((t.shape[0] / np.sqrt(np.pi) * a) * np.exp(-(t ** 2 / a ** 2)), i, -1)

    np.random.seed(0)
    X = X_ + 1 * np.random.randn(X_.shape[0], X_.shape[1], X_.shape[2]).astype(np.float32)

    y = np.zeros((length))
    y[0:int(length / 2)] = 1

    X = np.roll(X, int(length / 2), 0)

    if hard_case:
        # calculate the similarity matrix of original MiniROCKET transform encodings
        config = Config_orig()
        config.HDC_dim = 9996
        config.n_steps = length
        HDC_MINIROCKET = HDC_MINIROCKET_model(config)
        HDC_MINIROCKET.rocket = MiniRocketMultivariate(random_state=0)
        HDC_MINIROCKET.rocket.fit(X)
        X_MR = HDC_MINIROCKET.hdc_rocket_tf(X, scale=0)
        sim_mat_mr = cosine_similarity(X_MR, X_MR)

        # chose those indices with a high similarity between class one and two (empirically chosen)
        q = 0.96  # 96th quantile
        special_mat = sim_mat_mr[:250, 250:]
        thresh = np.quantile(special_mat, q)
        rows, cols = np.where((special_mat > thresh) & (special_mat < 1))
        cols = cols + 249
        idx = np.concatenate((np.unique(rows), np.unique(cols)))
        X_train, X_test, y_train, y_test = train_test_split(X[idx], y[idx], test_size=0.2, random_state=42)

    else:
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def df2np(X_train, X_test):
    '''
    convert pandas dataframe to numpy array
    @param X_train:
    @param X_test:
    @return:
    '''

    # train samples
    X = X_train
    num_samples = X.shape[0]
    num_dim = X.shape[1]
    s_length = X['dim_0'][0].size * 10 # set it bigger than necessary --> cut the tensor afterwards (to do so, we can handly different timeseries lengths)

    data_train = np.zeros([num_samples, num_dim, s_length])

    for s in range(num_samples):
        idx = 0
        for c in X.columns:
            series = X[c][s].values
            data_train[s, idx, :len(series)] = series
            idx += 1

    # test samples
    X = X_test
    num_samples = X.shape[0]
    num_dim = X.shape[1]

    data_test = np.zeros([num_samples, num_dim, s_length])

    for s in range(num_samples):
        idx = 0
        for c in X.columns:
            series = X[c][s].values
            data_test[s, idx, :len(series)] = series
            idx += 1

    # calculate the max length of all series
    data = np.concatenate([data_train, data_test],axis=0)

    # cut zeros from tensor
    S = np.sum(np.sum(data, axis=0), axis=0)
    z_idx = np.where(S == 0)

    X_train = data_train[:,:,:z_idx[0][0]]
    X_test = data_test[:, :, :z_idx[0][0]]

    return X_train, X_test
