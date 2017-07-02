import pickle as pkl
import numpy as np

K = 36

def open_pickle(file_name="train.pkl"):
    with open(file_name, 'rb') as f:
        return pkl.load(f)

def save_to_pickle(file_name, data_to_save):
    with open(file_name, 'wb') as f:
        pkl.dump(data_to_save, f)

def to_one_hot (number):
    one_hot = np.zeros(shape=(K,))
    one_hot[number[0]] = 1
    return one_hot

def one_hot(y):

    return np.array(list(map(to_one_hot, y)))

def group_data (x, y):
    N = x.shape[0]
    n1 = int(0.86*N)
    n2 = n1+ int(0.14*N)

    x_train = x[:n1]
    y_train = y[:n1]

    x_val = x[n1: n2 ]
    y_val = y[n1: n2]

    x_test = x[n2: ]
    y_test = y[n2: ]

    return x_train, y_train, x_val, y_val, x_test, y_test



