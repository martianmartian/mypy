import numpy as np
import matplotlib.pyplot as plt


def gauss_fun(x, u, sigma):
    delta = np.exp(-(x-u)**2/(2.0*sigma**2))
    return np.sqrt(2*np.pi*sigma**2)*delta


def poly_regression(x, count):
    data = []
    for i in range(count):
        data.append(np.power(x, i))
    return np.array(data)


def poly_weights(f, data):
    # f M*N   N: the count of data
    w_ml = np.dot(np.dot(np.linalg.inv(np.dot(f, f.T)), f), data)
    return w_ml


def poly_weights_normal(f, data, lanma):
    dim = f.shape[0]
    norm = lanma*np.identity(dim)
    w_ml = np.dot(np.dot(np.linalg.inv(norm+np.dot(f, f.T)), f), data)
    return w_ml


def gen_data(n=100, m=25):
    data = []
    loc = np.linspace(0, 1, 25)
    for i in range(100):
        sample = np.sin(2*np.pi*loc) + np.random.normal(0, 0.1, loc.shape[0])
        data.append(sample)
    return loc, np.array(data)


def predict(x, w, n):
    xw = poly_regression(x, n)
    y = np.dot(xw.T, w)
    return y


if __name__ == '__main__':
    # data = np.loadtxt('curvefitting')
    _x = np.linspace(0, 1, 10000)
    _y = np.sin(2*np.pi*_x)
    fig = plt.figure()
    plt.plot(_x, _y, '-g')
    count = 13
    loc, data = gen_data()
    plt.scatter(loc, data[0, :])
    w = []
    wf = poly_regression(loc, count)
    for i in range(data.shape[0]):
        w.append(poly_weights_normal(wf, data[i, :], 0.0000000003))
        # w.append(poly_weights_normal(wf, data[i, :], 0.000000000000000000000003))
        # w.append(poly_weights_normal(wf, data[i, :], 0.0000000000000000000000000000000003))
        # w.append(poly_weights_normal(wf, data[i, :], 0.000000000000000000000000000000000000000000003))
        # w.append(poly_weights(wf, data[i, :]))
    w = np.array(w)
    y = []
    mes = 0
    for i in range(w.shape[0]):
        y.append(predict(_x, w[i, :], count))
    y = np.array(y)
    yf = np.mean(y, axis=0)
    for i in range(23):
        plt.plot(_x, y[i, :], '-r')

    plt.plot(_x, yf, '-b')
    plt.show()
