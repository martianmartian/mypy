import numpy as np

class Tanh(object):
    def __init__(self):
        pass

    @staticmethod
    def function(x):
        return np.tanh( x )

    @staticmethod
    def derivative(x):
        return 1.0 - np.tanh(x)**2

    def __test(self):
        '''
        >>> x = np.array( [[-1, 3, -1, 1, 2], [1, -1, 0.5, -1, -2]] )
        >>> f = Tanh.function
        >>> y = f( x )
        >>> [['%.1f'%_ for _ in v] for v in y]
        [['-0.8', '1.0', '-0.8', '0.8', '1.0'], ['0.8', '-0.8', '0.5', '-0.8', '-1.0']]
        >>> d = Tanh.derivative
        >>> y = d( x )
        >>> [['%.1f'%_ for _ in v] for v in y]
        [['0.4', '0.0', '0.4', '0.4', '0.1'], ['0.4', '0.4', '0.8', '0.4', '0.1']]
        '''
        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
