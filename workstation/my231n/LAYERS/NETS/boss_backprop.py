#  there has to be a way to simplify these convoluted algo right?
import numpy as np
# import settings
from random import random, randint


def addends_matrix(a1, a2):
    lis = [0] * 14
    lis[a1 - 1] = 1 - 1
    lis[a1] = 1
    lis[a1 + 1] = 1 - 1
    lis[a2 + 6] = 1 - 1
    lis[a2 + 7] = 1
    lis[a2 + 8] = 1 - 1
    return lis

def sum_matrix(s):
    # lis = [0] * (13 + len(settings.strategies))
    lis = [0]*(10)
    lis[s] = 1
    return lis

def y_index(a1, a2):
    return 5 * (a1 - 1) + (a2 - 1)
    # 

class NeuralNetwork:

    def __init__(self, layers):
        self.err=[]
        self.activation = lambda x: np.tanh(x)
        self.activation_prime = lambda x: 1.0 - x**2
        self.weights = []
        self.sets=[]
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)  # prepared 3x3 for inputs
        r = 2*np.random.random((layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)  # prepared and appended 3X1 for next layer

    def fit(self, X, y, learning_rate=0.01, epochs=10000):

        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        for k in range(epochs):
            if len(self.err)!=0 and k>3000:
                if abs(self.err[-1])<0.01:
                    self.sets.append(self.weights)
                    break
            i = np.random.randint(X.shape[0])
            a = [X[i]]
            for l in range(len(self.weights)):
                dot_value = np.dot(a[l], self.weights[l]) 
                    # where did bias go?
                activation = self.activation(dot_value)
                a.append(activation)
            error = y[i] - a[-1]
            self.err.append(error[0])
            deltas = [error * self.activation_prime(a[-1])]
            for l in range(len(a) - 2, 0, -1):
                stp1=deltas[-1].dot(self.weights[l].T)
                deltas.append(stp1*self.activation_prime(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        # print "x",x
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=1)    
        # ones = np.atleast_2d(np.ones(x.shape[0]))
        # a = np.concatenate((ones.T, x), axis=1)
        # print "a",a
        for l in range(0, len(self.weights)):
            # print "self.weights",self.weights
            a = self.activation(np.dot(a, self.weights[l]))
            # print "a",a
        return a


if __name__ == '__main__':
    # main() is to learn to count 1,2,3,4,5
    # only for testing purposes.

    nn = NeuralNetwork([14,30,10])
    # X = np.array([[0, 0 ...],
    #               [0, 1 ...],
    #               [1, 0 ...],
    #               [1, 1 ...]])
    # y = np.array([0, 1, 1, 0])

    X = np.array([
        addends_matrix(1,0),
        addends_matrix(2,0),
        addends_matrix(3,0),
        addends_matrix(4,0),
        addends_matrix(5,0)
        ])
    y = np.array([
        sum_matrix(1),
        sum_matrix(2),
        sum_matrix(3),
        sum_matrix(4),
        sum_matrix(5)
        ])

    nn.fit(X, y)

    for e in X:
        print e,np.around(nn.predict(e), decimals=0)
        # print e==np.around(e, decimals=0)

    # import matplotlib.pyplot as plt
    # plt.plot(nn.err)
    # plt.show()

    # X1 = np.array([
    #     addends_matrix(1,2),
    #     addends_matrix(2,3),
    #     addends_matrix(3,4),
    #     addends_matrix(4,5)
    #     ])
    # y1 = np.array([
    #     sum_matrix(3),
    #     sum_matrix(4),
    #     sum_matrix(5),
    #     sum_matrix(6)
    #     ])

    # nn.err=[]
    # nn.fit(X1, y1)

    # # for e in X:
    # #     print(e,nn.predict(e))
    # import matplotlib.pyplot as plt
    # plt.plot(nn.err)
    # plt.show()


    # X2 = np.array([
    #     addends_matrix(1,0)
    #     ])

    # print nn.predict(X2)

    # X3 = np.array([
    #     addends_matrix(1,2)
    #     ])

    # print nn.predict(X3)
