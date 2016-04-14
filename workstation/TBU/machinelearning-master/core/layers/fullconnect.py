import math
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from nonlinears import Linear
from updaters.gradient_descent import GradientDescent

class Fullconnect(object):
    def __init__(self, input_size, output_size,
            nonlinear_function=Linear.function, derivative_function=Linear.derivative,
            updater=GradientDescent() ):
        self.input_size = input_size
        self.output_size = output_size
        # xavier initializer
        self.W = math.sqrt(6./(output_size+input_size)) * np.random.uniform( -1.0, 1.0, (input_size, output_size) )
        self.b = np.zeros( output_size )
        self.nonlinear_function = nonlinear_function
        self.derivative_function = derivative_function
        self.updater = updater

    def forward(self, x):
        self.x = np.copy(x)
        self.a = np.dot( self.x, self.W ) + self.b
        return self.nonlinear_function( self.a )

    def backward(self, delta):
        self.delta_a = delta * self.derivative_function(self.a)
        return np.dot( self.delta_a, self.W.T )

    def get_gradient(self):
        dW = np.dot( self.x.reshape( self.x.size/self.input_size, self.input_size ).T,
                self.delta_a.reshape( self.delta_a.size/self.output_size, self.output_size ) )
        db = self.delta_a.reshape( self.delta_a.size/self.output_size, self.output_size ).sum(axis=0)
        return (dW, db)

    def update(self):
        dW, db = self.get_gradient()
        self.W = self.updater.update(self.W, dW)
        self.b = self.updater.update(self.b, db)

    def folk(self, shared=False):
        _ = Fullconnect(self.input_size, self.output_size,
                self.nonlinear_function, self.derivative_function,
                updater=GradientDescent())
        if shared:
            _.W = self.W
            _.b = self.b
        return _

    def __test(self):
        '''
        >>> x = np.array([1, 2, 3])
        >>> l = Fullconnect(3, 4)
        >>> l.W = np.eye(3, 4)
        >>> l.b = np.array([0.3, 0.5, 0, 0])
        >>> y = l.forward( x )
        >>> y.shape
        (4,)
        >>> print [_ for _ in y]
        [1.3, 2.5, 3.0, 0.0]
        >>> delta = np.array([1, 1, 1, 1])
        >>> d = l.backward( delta )
        >>> print  [_ for _ in d]
        [1.0, 1.0, 1.0]
        >>> x = np.array([[1, 2, 3], [2, 3, 4]])
        >>> y = l.forward( x )
        >>> y.shape
        (2, 4)
        >>> delta = np.array([[1,1, 1, 1], [2, 2, 2, 2]])
        >>> d = l.backward( delta )
        >>> x.shape == d.shape
        True
        >>> dW, db = l.get_gradient()
        >>> dW.shape
        (3, 4)
        >>> db.shape
        (4,)
        >>> l.update()
        >>> x = np.array([[[1, 2, 3], [2, 3, 4]]] * 3)
        >>> y = l.forward( x )
        >>> y.shape
        (3, 2, 4)
        '''
        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
