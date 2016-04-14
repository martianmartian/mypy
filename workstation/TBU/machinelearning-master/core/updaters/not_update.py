import math
import numpy as np

class NotUpdate:
    def __init__(self, learning_rate=0.01):
        pass

    def update(self, param, gradient):
        return param

    def __test(self):
        '''
        >>> u = NotUpdate()
        >>> x = np.array( [[1, 2, 3], [2, 3, 4]], dtype=float )
        >>> grad = np.array( [[1, 2, 3], [2, 3, 4]], dtype=float )
        >>> x = u.update( x, grad )
        >>> [['%.2f'%_ for _ in v] for v in x]
        [['1.00', '2.00', '3.00'], ['2.00', '3.00', '4.00']]
        '''
        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
