import numpy as np
import matplotlib.pyplot as pl
# import matplotlib.pyplot as plt

from numpy.random import multivariate_normal as mvg

class Collection:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)

class Kernel:
    def __init__(self, name, parameters={}):
        def rbf(x, y):
            l = 1
            if 'l' in parameters:
                l = parameters['l']
            return exp(-1.0/(2*l*l)*np.linalg.norm(x - y)**2)
        def dot(x, y):
            return np.dot(x, y) + 1
        funcs = {
            'rbf' : rbf,
            'dot' : dot
        }
        self.__call__ = funcs[name]


class GP:
    def __init__(self, m, k, parameters={}):
        self.m = m
        self.k = k
        self.num_points = 0
        self.parameters = parameters

    def sample(self, points, num=1):
        n = len(points)
        if self.num_points > 0:
            kyy = self.gen_covmat(points, points)
            kyx = self.gen_covmat(points, self.x)   # points is the point where we will evaulate its vaule y is row x is col
            kxy = self.gen_covmat(self.x, points)
            kxx = np.linalg.inv(self.kxx)
            covmat = kyy - kyx*kxx*kxy
            y = np.transpose([self.y])
            mean = kyx*kxx*y
            mean = np.array(np.transpose(mean))[0]
        else:
            covmat = self.gen_covmat(points, points)
            mean = np.zeros(n)
        return mvg(mean, covmat, num)

    def gen_covmat(self, m1, m2):
        return np.matrix([[self.k(x, y) for x in m2] for y in m1])

    def train(self, collection):
        self.x, self.y = collection.x, collection.y
        self.num_points = len(self.x)
        if 'noise' in self.parameters:
            noise = np.identity(len(self.x)) * self.parameters['noise']
        else:
            noise = np.zeros((len(self.x), len(self.x)))
        self.kxx = self.gen_covmat(self.x, self.x) + noise

if __name__ == "__main__":
    # mean = [0, 0]
    # cov = [[1, 0], [0, 100]]  # diagonal covariance
    # x, y = np.random.multivariate_normal(mean, cov, 5000).T
    # print x.shape
    # print y.shape
    # plt.plot(x, y, 'x')
    # plt.axis('equal')
    # plt.show()

    from math import exp
    def m(x):
        return 0
    gp = GP(m, Kernel('rbf', parameters={'l' : 2}), parameters={
        "noise" : 0.01,
    })
    train = Collection([1, 2, 3, 4],[1, 4, 9, 16])
    train = Collection([0.00361157, 0.02356921, 0.09201148, 0.11695927, 0.26331115, 0.26627008, 0.34690724, 0.35863524, 0.37182364, 0.40207368, 0.51940526, 0.52171434,
                0.52753526,  0.64063631,  0.64819143,  0.66338127,  0.69000325,  0.72323216, 0.74251518,  0.74467774,  0.75149699,  0.7632613,   0.78782982,  0.79736613,
                0.81658858,  0.82105877,  0.83103044,  0.84143118,  0.87071978,  0.87135029, 0.89197245,  0.93548946,  0.99502295],
            [-0.47933465, 0.24601366,  0.49620931,  0.0280998, 0.74705795, 0.41211598, 0.11280758, 0.89295853, 0.86512994, 0.63444975, -0.28111649, -0.52089386,0.1356233,
             -0.84391977, -0.51340666, -0.70227695, -0.79910531, -0.42563439, -0.27232769, -0.36675029, -0.98348514, -1.35867041, -0.4292669,  -0.89799212, -1.36289829,
             -0.92174124, -0.64918582, -0.74471005, -0.62116379, -0.8963551, -0.98623757, -0.36044488, 0.21837434])
    #train = Collection([0],[0])
    gp.train(train)
    x = np.arange(-5, 10, 0.05)
    points = np.array(x)
    samples = gp.sample(points, 100)
    [pl.plot(x, samples[i]) for i in range(len(samples))]
    pl.plot(train.x, train.y, 'ro')
    pl.show()