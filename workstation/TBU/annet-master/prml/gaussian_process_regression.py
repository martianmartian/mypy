import numpy as np
import matplotlib.pyplot as plt
from math import exp


def generate_data_regression():
    x = np.random.uniform(0, 1, 7)
    noise = np.random.normal(0, 0.3, 7)
    real_x = np.linspace(0, 1, 1000)
    x.sort()
    y = np.sin(x*2*np.pi)
    y += noise
    real_y = np.sin(real_x*2*np.pi)
    plt.plot(real_x, real_y, 'g-')
    plt.scatter(x, y, color='r', marker='*')
    # plt.show()
    return x, y


def generate_data_classification():
    b0 = 2*np.random.normal(0, 0.3, 100)
    b1 = 2*np.random.normal(0, 0.4, 100)
    plt.scatter(b0, b1, color='b', marker='o', alpha=0.4)
    r0 = np.random.normal(1, 0.3, 100)
    r1 = np.random.normal(1, 0.5, 50)
    r2 = np.random.normal(-1, 0.5, 50)
    plt.scatter(r0[:50], r1, color='red', marker='*', alpha=0.4)
    plt.scatter(r0[50:], r2, color='red', marker='*', alpha=0.4)
    # r0 = 2*np.random.normal()

    plt.show()


def kernel(x, y, l=2.0):
    # d = (x - y)**2 / sigma
    # return np.exp(-0.5 * d)
    return exp(-1.0/(2*l*l)*np.linalg.norm(x - y)**2)


def gram_matrix(x, y):
    return np.array([[kernel(i, j) for j in x] for i in y])


class Collections(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Gaussian(object):
    def __init__(self, kn, x, y, points, beta=0.1):
        self.kn = kn
        self.beta = beta
        self.x = x
        self.y = y
        self.points = points

    def train(self):
        noise = np.identity(len(self.x))*self.beta
        self.gram = self.gram_matrix(self.x, self.x)+noise

    def regression(self):
        kpp = self.gram_matrix(self.points, self.points)
        kpx = self.gram_matrix(self.x, self.points)
        kpy = self.gram_matrix(self.points, self.x)
        inv_gram = np.linalg.inv(self.gram)
        cov = kpp - np.dot(np.dot(kpx, inv_gram), kpy)
        print cov.shape
        mean = np.dot(np.dot(kpx, inv_gram), self.y)
        print mean.shape
        return np.random.multivariate_normal(mean, cov, 100), mean

    def gram_matrix(self, x, y):
        return np.array([[self.kn(i, j) for j in x] for i in y])

def sample_no_prior():
    z = np.linspace(0, 1, 1000)
    gram_z = gram_matrix(z, z)
    mean = np.zeros(1000)
    datas = np.random.multivariate_normal(mean, gram_z, 100)
    for data in datas:
        plt.plot(z, data)
    plt.show()


if __name__ == '__main__':
    x, y = generate_data_regression()
    z = np.linspace(0, 1, 1000)
    gp = Gaussian(kernel, x, y, z)
    gp.train()
    datas, m = gp.regression()
    for data in datas:
        plt.plot(z, data)
    plt.plot(z, m)
    plt.show()





