import numpy as np
from utils import save_to_pickle

# def soft_max (w, x_train):
#     """
#     :param w: macierz parametrów wymiaru KxD
#     :param x_train: macierz zawierająca obserwacje NxD
#     :return : macierz teta NxK czymkolwiek ona tam jest
#     """
#     licznik = np.exp(x_train @ w.T)
#
#     n, _= licznik.shape
#     mianownik = np.sum(licznik, axis=1)
#     return np.divide(licznik, mianownik.reshape(n,1))

def soft_max(w, x):
    z = x@w.T
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div

def logistic_cost_function(w, x_train, y_train):
    '''
    :param w: parametry modelu KxD
    :param x_train: ciag treningowy - wejscia NxD
    :param y_train: ciag treningowy - wyjscia Nx1
    :return: funkcja zwraca gradient funkcji logistycznej po w
    '''

    theta = soft_max(w, x_train)
    return (y_train - theta).T @ x_train


def regularized_logistic_cost_function(w, x_train, y_train, regularization_lambda):
    '''
    :param w: parametry modelu Mx1
    :param x_train: ciag treningowy - wejscia NxM
    :param y_train: ciag treningowy - wyjscia Nx1
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca gradient funkcji logistycznej z regularyzacją po w
    '''

    grad = logistic_cost_function(w, x_train, y_train)
    grad[1:] += regularization_lambda * w[1:]

    return np.array(grad)

def stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch):
    '''
    :param obj_fun: funkcja celu, ktora ma byc optymalizowana.
    :param x_train: dane treningowe wejsciowe NxD
    :param y_train: dane treningowe wyjsciowe NxK
    :param w0: punkt startowy KxD
    :param epochs: liczba epok
    :param eta: krok uczenia
    :param mini_batch: wielkosc mini-batcha
    :return: funkcja wykonuje optymalizacje metoda stochastycznego gradientu prostego dla funkcji obj_fun. Zwraca znaleziony optymalny punkt w.
    '''

    batchX = []
    batchY = []

    x = x_train
    y = y_train

    while len(x) > 0:
        batchX.append(x[:mini_batch])
        x = x[mini_batch:]
        batchY.append(y[:mini_batch])
        y = y[mini_batch:]

    w = w0

    batch = list(zip(batchX, batchY))

    for k in range(epochs):
        for mx, my in batch:
            grad = obj_fun(w, mx, my)
            w = w + eta * grad

    return w

def model_selection(x_train, y_train, x_val, y_val, w0, epochs, eta, mini_batch, lambdas):
    '''
    :param x_train: ciag treningowy wejsciowy NxD
    :param y_train: ciag treningowy wyjsciowy NxK
    :param x_val: ciag walidacyjny wejsciowy Nval x D
    :param y_val: ciag walidacyjny wyjsciowy Nval x K
    :param w0: wektor poczatkowych wartosci parametrow
    :param epochs: liczba epok dla SGD
    :param eta: kroki uczenia, które maja byc sprawdzone
    :param mini_batch: wielkosci mini batcha, ktore maja byc sprawdzone
    :param lambdas: lista wartosci parametru regularyzacji lambda, ktore maja byc sprawdzone
    :return: funkcja wykonuje selekcje modelu. Zwraca krotke (best_lambda, best_w, best_error, best_eta, best_epochs, best_mb),
            która przedstawia wartości parametrów dla najlepszego wybranego modelu.
    '''

    best_lambda = 0
    best_w = w0
    best_error = 1
    best_eta = 0
    best_mb = 0
    best_epochs = 0

    for current_lamda in lambdas:
        for current_epochs in epochs:
            for current_eta in eta:
                for current_mb in mini_batch:
                    def nowa(w, x, y):
                        return regularized_logistic_cost_function(w, x, y, current_lamda)
                    w = stochastic_gradient_descent(nowa, x_train, y_train, w0, current_epochs, current_eta, current_mb)
                    y_pred = prediction(x_val, w)
                    current_error = prediction_error(y_pred, y_val)
                    print("Lambda: {}, Eta: {}, Batch: {}, Epochs: {} → ERROR: {}".format(
                        current_lamda, current_eta, current_mb, current_epochs, current_error))
                    if (current_error < best_error):
                        best_lambda = current_lamda
                        best_w = w
                        best_error = current_error
                        best_eta = current_eta
                        best_mb = current_mb
                        best_epochs = current_epochs
                        save_to_pickle('parameters-thebest.pkl', best_w)


    return best_lambda, best_w, best_error, best_eta, best_epochs, best_mb



def prediction (x, w):
    """
    :param x: obserwacje NxD
    :param w: parametry KxD
    :return: przewidywane klasy Nx1
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

def prediction_error (predict_y, true_y):
    error = 0
    N = len(predict_y)
    for i in range(N):
        error += predict_y[i,0] != true_y[i,0]

    return error / N