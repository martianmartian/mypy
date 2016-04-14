import numpy as np

class Ramp(object):
    def __init__(self):
        pass

    @staticmethod
    def function(x):
        return np.maximum( -1.0, np.minimum( 1.0, x ) )

    @staticmethod
    def derivative(x):
        _ = x.reshape( (np.prod(x.shape), ) )
        return np.array( [(1.0 if( v>=-1.0 and v<=1.0 ) else 0.0) for v in _] ).reshape( x.shape )

    def __test(self):
        '''
        >>> x = np.array( [[-1, 3, -1, 1, 2], [1, -1, 0.5, -1, -2]] )
        >>> f = Ramp.function
        >>> y = f( x )
        >>> [['%.1f'%_ for _ in v] for v in y]
        [['-1.0', '1.0', '-1.0', '1.0', '1.0'], ['1.0', '-1.0', '0.5', '-1.0', '-1.0']]
        >>> d = Ramp.derivative
        >>> y = d( x )
        >>> [['%.1f'%_ for _ in v] for v in y]
        [['1.0', '0.0', '1.0', '1.0', '0.0'], ['1.0', '1.0', '1.0', '1.0', '0.0']]
        '''
        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
