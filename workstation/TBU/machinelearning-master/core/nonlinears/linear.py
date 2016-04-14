import numpy as np

class Linear(object):
    def __init__(self):
        pass

    @staticmethod
    def function(x):
        return x

    @staticmethod
    def derivative(x):
        return np.ones(x.shape)

    def __test(self):
        '''
        >>> x = np.array( [[-1, 3, -1, 1, 2], [1, -1, 0.5, -1, -2]] )
        >>> f = Linear.function
        >>> y = f( x )
        >>> [['%.1f'%_ for _ in v] for v in y]
        [['-1.0', '3.0', '-1.0', '1.0', '2.0'], ['1.0', '-1.0', '0.5', '-1.0', '-2.0']]
        >>> d = Linear.derivative
        >>> y = d( x )
        >>> [['%.1f'%_ for _ in v] for v in y]
        [['1.0', '1.0', '1.0', '1.0', '1.0'], ['1.0', '1.0', '1.0', '1.0', '1.0']]
        '''
        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
