import math
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from nonlinears import Linear
from updaters.gradient_descent import GradientDescent
from layers import Fullconnect

class Dropout(Fullconnect):
    def __init__(self, input_size, output_size, drop_ratio=0.5,
            nonlinear_function=Linear.function, derivative_function=Linear.derivative,
            updater=GradientDescent() ):
        super(Dropout, self).__init__(input_size, output_size, nonlinear_function, derivative_function, updater)
        self.drop_ratio = drop_ratio
        self.is_testing = False

    def forward(self, x):
        a = super(Dropout, self).forward(x)
        if not self.is_testing:
            self.drop_map = np.array( [1.0 if v>=self.drop_ratio else 0.0 for v in np.random.uniform(0, 1, np.prod( a.shape ))] ).reshape( a.shape )
            self.a = np.multiply(a, self.drop_map) * (1.0 / self.drop_ratio)
        return self.nonlinear_function( self.a )

    def backward(self, delta):
        return super(Dropout, self).backward( np.multiply(delta, self.drop_map) )

    def __test(self):
        '''
        >>> np.random.seed(0xC0FFEE)
        >>> x = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
        >>> x.shape
        (2, 2, 3)
        >>> l = Dropout(3, 4)
        >>> y = l.forward( x )
        >>> y.shape
        (2, 2, 4)
        >>> print ['%.1f'%_ for _ in y[0][0]]
        ['0.0', '5.5', '1.2', '-7.2']
        >>> np.array_equal( y, l.forward( x ) )
        False
        >>> delta = np.array([[[1,1,1,1], [1,1,1,1]], [[0,0,0,0], [2,2,2,2]]])
        >>> d = l.backward( delta )
        >>> print ['%.1f'%_ for _ in d[0][0]]
        ['-0.2', '-0.1', '-0.2']
        >>> x.shape == d.shape
        True
        >>> l.update()
        >>> type( l ).__name__
        'Dropout'
        '''
        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
