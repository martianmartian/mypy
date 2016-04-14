import numpy as np

class ReLU(object):
    def __init__(self):
        pass

    @staticmethod
    def function(x):
        mapper = np.zeros_like( x )
        return np.fmax( x, mapper )

    @staticmethod
    def derivative(x):
        _ = x.reshape( (np.prod(x.shape), ) )
        return np.array( [(1.0 if v>0 else 0.0) for v in _] ).reshape( x.shape )

    def __test(self):
        '''
        >>> x = np.array( [[-1, 3, -1, 1, 2], [1, -1, 0.5, -1, -2]] )
        >>> f = ReLU.function
        >>> y = f( x )
        >>> [['%.1f'%_ for _ in v] for v in y]
        [['0.0', '3.0', '0.0', '1.0', '2.0'], ['1.0', '0.0', '0.5', '0.0', '0.0']]
        >>> d = ReLU.derivative
        >>> y = d( x )
        >>> [['%.1f'%_ for _ in v] for v in y]
        [['0.0', '1.0', '0.0', '1.0', '1.0'], ['1.0', '0.0', '1.0', '0.0', '0.0']]
        '''
        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
