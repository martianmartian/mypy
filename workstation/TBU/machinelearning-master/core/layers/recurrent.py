import math
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from nonlinears import Linear
from updaters.gradient_descent import GradientDescent
from layers import Fullconnect

class Recurrent(Fullconnect):
    def __init__(self, input_size, output_size,
            nonlinear_function=Linear.function, derivative_function=Linear.derivative,
            updater=GradientDescent() ):
        super(Recurrent, self).__init__(input_size+output_size, output_size, nonlinear_function, derivative_function, updater)

    def forward(self, x):
        time_splited_x = [_ for _ in np.split( x, x.shape[0] )]
        self.times = len(time_splited_x)

        shape = list(time_splited_x[0].shape)
        shape[-1] = self.output_size

        self.shared_layers = []
        outputs = [np.array([])] * (self.times+1)
        outputs[-1] = np.zeros(shape)
        for t, splited in enumerate(time_splited_x):
            layer = self.folk(shared=True)
            outputs[t] = layer.forward( np.concatenate( [splited, outputs[t-1]], axis=-1 ) )
            self.shared_layers.append( layer )

        return np.concatenate( outputs[:self.times] )

    def backward(self, delta):
        time_splited_delta = [_ for _ in np.split( delta, delta.shape[0] )]
        self.times = len(time_splited_delta)

        shape = list(time_splited_delta[0].shape)
        shape[-1] = self.output_size

        deltas = [np.array([])] * (self.times+1)
        deltas[-1] = np.zeros(shape)
        outputs = []
        for t, layer, splited in reversed( zip(range(self.times), self.shared_layers, time_splited_delta) ):
            _, deltas[t] = np.split( layer.backward( splited+deltas[t+1] ), [self.input_size-self.output_size], axis=-1 )
            outputs.append( _ )

        return np.concatenate( outputs[:self.times] )

    def update(self):
        dW = np.zeros_like( self.W )
        db = np.zeros_like( self.b )
        for layer in self.shared_layers:
            _ = layer.get_gradient()
            dW += _[0]
            db += _[1]
        self.W = self.updater.update(self.W, dW)
        self.b = self.updater.update(self.b, db)

    def __test(self):
        '''
        >>> np.random.seed(0xC0FFEE)
        >>> x = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
        >>> x.shape
        (2, 2, 3)
        >>> l = Recurrent(3, 4)
        >>> y = l.forward( x )
        >>> y.shape
        (2, 2, 4)
        >>> print ['%.1f'%_ for _ in y[0][0]]
        ['1.0', '2.2', '0.5', '-2.9']
        >>> np.array_equal( y, l.forward( x ) )
        True
        >>> delta = np.array([[[1,1,1,1], [1,1,1,1]], [[0,0,0,0], [2,2,2,2]]])
        >>> d = l.backward( delta )
        >>> x.shape == d.shape
        True
        >>> l.update()
        '''
        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
