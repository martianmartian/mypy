
import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2


class NeuralNetwork:

    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime

        self.weights = []
        for i in range(1, len(layers) - 1): # [2,2,1], 1 loop, i = 1
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            print r
            self.weights.append(r)  # prepared 3x3 for inputs
        r = 2*np.random.random((layers[i] + 1, layers[i+1])) - 1   #i = 1
        self.weights.append(r)  # prepared and appended 3X1 for next layer
        print self.weights

    def fit(self, X, y, learning_rate=0.2, epochs=20):
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        # recreated input stack X with addition of biases.
        # now it's 4x3
        for k in range(epochs):
            # if k % 10000 == 0: print 'epochs:', k

            i = np.random.randint(X.shape[0])
            # choosing one random type of input

            a = [X[i]]  
            # [[1,1,1]]=>[[1,1,1],[1,0,1]]=>[[1,1,1],[1,0,1],[0,0,1]]=>[[1,1,1],[1,0,1],[0,0,1],[0,0,1]]
            for l in range(len(self.weights)):
                dot_value = np.dot(a[l], self.weights[l]) 

                activation = self.activation(dot_value)
                # """activation == charge"""
                a.append(activation)  


            #a, deltas :  1x3
            error = y[i] - a[-1]  # a[-1] = [0,0,1]
            deltas = [error * self.activation_prime(a[-1])]
            # initialized deltas =[[0.1,0.3,0.4]], just like a[X[i]] to begin with

            # we need to begin at the second to last layer 
            # (a layer before the output layer)
            for l in range(len(a) - 2, 0, -1):
                stp1=deltas[-1].dot(self.weights[l].T)
                deltas.append(stp1*self.activation_prime(a[l]))

            # reverse
            # [level3(output)->level2(hidden)]  
            #   => [level2(hidden)->level3(output)]
            deltas.reverse()

            # backpropagation
            # 1. Multiply its output delta and input activation 
            #    to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x): 
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=1)      
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

if __name__ == '__main__':

    nn = NeuralNetwork([2,2,1])

    # X = np.array([[0, 0],
    #               [0, 1],
    #               [1, 0],
    #               [1, 1]])

    # y = np.array([0, 1, 1, 0])


    X = np.array([[1, 1]])

    y = np.array([0])

    nn.fit(X, y)

    for e in X:
        print(e,nn.predict(e))


