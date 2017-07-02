from hog import extract_all
import pickle as pkl
import numpy as np

def open_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pkl.load(f)

def predict(x):
    w = open_pickle('parameters-thebest.pkl')
    extracted = extract_all(x)
    y_pred = prediction(extracted, w)

    return y_pred

def prediction (x, w):
    """
    :param x: obserwacje NxD
    :param w: parametry KxD
    :return:
    """
    N = x.shape[0]
    z = x @ w.T
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    licznik = np.exp(z - s)
    y_pred = []
    for n in range(N):
        y_pred.append([np.argmax(licznik[n])])

    return np.array(y_pred)





