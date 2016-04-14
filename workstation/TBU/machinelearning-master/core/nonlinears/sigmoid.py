import numpy as np

class Sigmoid(object):
    def __init__(self):
        pass

    @staticmethod
    def function(x):
        return 1. / (1. + np.exp(-x))

    @staticmethod
    def derivative(x):
        return Sigmoid.function(x) * Sigmoid.function(1-x)

    def __test(self):
        '''
        >>> x = np.array( [[-1, 3, -1, 1, 2], [1, -1, 0.5, -1, -2]] )
        >>> f = Sigmoid.function
        >>> y = f( x )
        >>> [['%.2f'%_ for _ in v] for v in y]
        [['0.27', '0.95', '0.27', '0.73', '0.88'], ['0.73', '0.27', '0.62', '0.27', '0.12']]
        >>> d = Sigmoid.derivative
        >>> y = d( x )
        >>> [['%.2f'%_ for _ in v] for v in y]
        [['0.24', '0.11', '0.24', '0.37', '0.24'], ['0.37', '0.24', '0.39', '0.24', '0.11']]
        '''
        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
