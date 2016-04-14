import math
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from nonlinears import Linear
from layers import Fullconnect
from updaters import NotUpdate

class Bypass(Fullconnect):
    def __init__(self, input_size,
            nonlinear_function=Linear.function, derivative_function=Linear.derivative ):
        super(Bypass, self).__init__(input_size, input_size, nonlinear_function, derivative_function, NotUpdate())
        self.W = np.identity( input_size )

    def __test(self):
        '''
        >>> x = np.array([1, 2, 3])
        >>> l = Bypass(3)
        >>> y = l.forward( x )
        >>> y.shape
        (3,)
        >>> print [_ for _ in y]
        [1.0, 2.0, 3.0]
        >>> delta = np.array([1, 1, 1])
        >>> d = l.backward( delta )
        >>> print  [_ for _ in d]
        [1.0, 1.0, 1.0]
        >>> x = np.array([[1, 2, 3], [2, 3, 4]])
        >>> y = l.forward( x )
        >>> y.shape
        (2, 3)
        >>> delta = np.array([[1,1, 1], [2, 2, 2]])
        >>> d = l.backward( delta )
        >>> x.shape == d.shape
        True
        >>> dW, db = l.get_gradient()
        >>> dW.shape
        (3, 3)
        >>> db.shape
        (3,)
        >>> l.update()
        >>> x = np.array([[[1, 2, 3], [2, 3, 4]]] * 3)
        >>> y = l.forward( x )
        >>> y.shape
        (3, 2, 3)
        '''
        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
