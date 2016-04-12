import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab


# reference  https://github.com/jamt9000/prml/blob/master/3.3.1-parameter-distribution.ipynb

def gen_data():
    x1 = np.random.uniform(-1, 1, 20)
    x0 = np.ones((20, 1))
    a, b = -0.3, 0.5
    y = a + b*x1
    y = y + np.random.normal(0, 0.04, 20)
    x1 = x1[:, np.newaxis]
    _x = np.hstack([x0, x1])
    return x1, y, _x


def gaussian(x, u, sigma):
    demn = np.sqrt(2*np.pi)*sigma
    dima = np.exp(-1*np.power(x-u, 2)/(2*sigma**2))
    return dima/demn


def gaussian2(w0, w1, x, y, sigma):
    z = w0 + w1*x
    demn = np.sqrt(2*np.pi)*sigma
    dima = np.exp(-1*np.power(z-y, 2)/(2*sigma**2))
    return dima/demn


def draw_sample(u, sigma):
    values = []
    flag = True
    while len(values) < 6:
        flag = True
        while flag:
            x1, x2 = np.random.uniform(-1, 1, 2)
            value = gaussian(x1, u, sigma)*gaussian(x2, u, sigma)
            if value > 0.3:
                values.append([x1, x2])
                flag = False
    return values


def caculate_posterior(x, y, a, b):
    f = np.array([1, x])
    f = f[np.newaxis, :]
    ff = np.dot(f.T, f)
    _a = a*np.eye(2, 2)
    s_n = _a + b*ff
    _s_n = np.linalg.inv(s_n)
    u = b*np.dot(_s_n, f.T*y)
    return u, _s_n


def caculate_posterior_dim(x, y, a, b):
    s_n = a*np.eye(2) + b*np.dot(x.T, x)
    _s_n = np.linalg.inv(s_n)
    u = b*np.dot(_s_n, np.dot(x.T, y))
    return u, _s_n


def mult_gaussian(w, u, sigma):
    demn = 2*np.pi*np.sqrt(np.linalg.det(sigma))
    u1 = w-u
    sigma = np.linalg.inv(sigma)
    delt = np.dot(np.dot(u1.T, sigma), u1)
    dima = np.exp(-0.5*delt)
    return dima/demn


if __name__ == '__main__':
    x, y, all_x = gen_data()
    _x = np.linspace(1, -1, 1000)
    _y = -0.3 + 0.5*_x
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(4, 3, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    ax1.plot(_x, _y, 'r')
    ax1.scatter(x, y)

    prior_dist_w0 = np.linspace(1, -1, 1000)
    prior_w0, prior_w1 = np.meshgrid(prior_dist_w0, prior_dist_w0)
    result = gaussian(prior_w0, 0, 0.5)*gaussian(prior_w1, 0, 0.5)
    resultss = mlab.bivariate_normal(prior_w0, prior_w1, sigmax=0.5, sigmay=0.5, mux=0, muy=0, sigmaxy=0)
    ax2.contourf(prior_w0, prior_w1, result)

    result3 = draw_sample(0, 0.2)
    for (x0, x1) in result3:
        _y = x0 + x1*_x
        ax3.plot(_x, _y, 'r')

    result4 = gaussian2(prior_w0, prior_w1, x[0], y[0], 0.2)
    ax4.contourf(prior_w0, prior_w1, result4)

    result5 = np.zeros((prior_w0.shape[0], prior_w0.shape[1]))
    u, sigma = caculate_posterior(x[0], y[0], 2.0, 25)

    Z = mlab.bivariate_normal(prior_w0, prior_w1, sigmax=np.sqrt(sigma[0, 0]), sigmay=np.sqrt(sigma[1, 1]), mux=u[0],
                              muy=u[1], sigmaxy=sigma[0, 1])
    ax5.contourf(prior_w0, prior_w1, Z)
    u = u.reshape((1, 2))
    W = np.random.multivariate_normal(u[0], sigma, 6)
    for w in W:
        ax6.plot(_x, w[0]+w[1]*_x)
    ax6.scatter(x[0], y[0])

    result7 = gaussian2(prior_w0, prior_w1, x[1], y[1], 0.2)
    ax7.contourf(prior_w0, prior_w1, result7)

    u, sigma = caculate_posterior_dim(all_x[:2, :], y[:2], 2, 25)
    Z = mlab.bivariate_normal(prior_w0, prior_w1, sigmax=np.sqrt(sigma[0, 0]), sigmay=np.sqrt(sigma[1, 1]),
                              mux=u[0], muy=u[1], sigmaxy=sigma[0, 1])
    ax8.contourf(prior_w0, prior_w1, Z)
    W = np.random.multivariate_normal(u.reshape((1, 2))[0], sigma, 6)
    for w in W:
        ax9.plot(_x, w[0]+w[1]*_x)
    ax9.scatter(x[:2], y[:2])

    result10 = gaussian2(prior_w0, prior_w1, x[19], y[19], 0.2)
    ax10.contourf(prior_w0, prior_w1, result10)

    u, sigma = caculate_posterior_dim(all_x[:20, :], y[:20], 2, 25)
    Z = mlab.bivariate_normal(prior_w0, prior_w1, sigmax=np.sqrt(sigma[0, 0]), sigmay=np.sqrt(sigma[1, 1]),
                              mux=u[0], muy=u[1], sigmaxy=sigma[0, 1])
    ax11.contourf(prior_w0, prior_w1, Z)

    W = np.random.multivariate_normal(u.reshape((1, 2))[0], sigma, 6)
    for w in W:
        ax12.plot(_x, w[0]+w[1]*_x)
    ax12.scatter(x[:20], y[:20])
    plt.show()

