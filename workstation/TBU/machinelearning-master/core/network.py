import numpy as np
from activations import Softmax, Sigmoid, Identity

class Network(object):
    def __init__(self):
        self.layers = []
        self.activation = Softmax()
        self.init()

    def init(self):
        for layer in self.layers:
            layer.init()

    def predict(self, x):
        _ = x
        for layer in self.layers:
            _ = layer.forward( _ )
        self.y = self.activation.forward( _ )
        return self.y

    def train(self, x, target):
        y = self.predict( x )
        _ = self.activation.backward( y, target )
        for layer in reversed( self.layers ):
            _ = layer.backward( _ )
            layer.update()
        return self.activation.loss( y, target )

    def dump_params(self):
        odict = {}
        odict['layers'] = [(l.W, l.b) for l in self.layers]
        return odict

    def load_params(self, idict):
        for l, p in zip(self.layers, idict['layers']):
            l.W, l.b= p

    def __test(self):
        '''
        >>> from layers import Fullconnect
        >>> from nonlinears import ReLU, Tanh
        >>> from activations import Softmax, Sigmoid, Identity
        >>> from updaters import GradientDescent, NotUpdate
        >>>
        >>> learning_rate = 0.01
        >>> np.random.seed(0xC0FFEE)
        >>>
        >>> # Multiclass classification
        >>> n = Network()
        >>> n.layers.append( Fullconnect(2, 10, ReLU.function, ReLU.derivative, updater=GradientDescent(learning_rate)) )
        >>> n.layers.append( Fullconnect(10, 2, updater=GradientDescent(learning_rate)) )
        >>> n.activation = Softmax()
        >>>
        >>> for epoch in range(0, 20):
        ...     loss = n.train( x = np.array([ [1, 2, 1, 2,  5, 6, 5, 6],
        ...                                    [5, 4, 4, 5,  1, 2, 2, 1]]).T,
        ...                target = np.array([ [1, 1, 1, 1,  0, 0, 0, 0],
        ...                                    [0, 0, 0, 0,  1, 1, 1, 1]]).T )
        ...     if epoch%5 == 0:
        ...         print 'epoch:%04d loss:%.2f'%(epoch, loss)
        epoch:0000 loss:6.64
        epoch:0005 loss:0.65
        epoch:0010 loss:0.36
        epoch:0015 loss:0.25
        >>>
        >>> y = n.predict( np.array( [[1, 6, 3], [5, 1, 4]] ).T )
        >>> print ['%.2f'%_ for _ in y[0]]
        ['0.99', '0.01']
        >>> [_ for _ in np.argmax(y, -1)]
        [0, 1, 0]
        >>>
        >>> # Multiple-class classification
        >>> n = Network()
        >>> n.layers.append( Fullconnect(2, 10, ReLU.function, ReLU.derivative, updater=GradientDescent(learning_rate)) )
        >>> n.layers.append( Fullconnect(10, 2, updater=GradientDescent(learning_rate)) )
        >>> n.activation = Sigmoid()
        >>>
        >>> for epoch in range(0, 20):
        ...     loss = n.train( x = np.array([ [1, 2, 1, 2,  4, 5, 4, 5,  5, 6, 5, 6],
        ...                                    [5, 4, 4, 5,  5, 4, 5, 4,  1, 2, 2, 1]]).T,
        ...                target = np.array([ [1, 1, 1, 1,  1, 1, 1, 1,  0, 0, 0, 0],
        ...                                    [0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 1, 1]]).T )
        ...     if epoch%5 == 0:
        ...         print 'epoch:%04d loss:%.2f'%(epoch, loss)
        epoch:0000 loss:29.39
        epoch:0005 loss:4.78
        epoch:0010 loss:3.38
        epoch:0015 loss:2.64
        >>>
        >>> y = n.predict( np.array( [[5, 6, 3], [4, 1, 4]] ).T )
        >>> print ['%.2f'%_ for _ in y[0]]
        ['0.86', '0.96']
        >>>
        >>> # Regression
        >>> n = Network()
        >>> n.layers.append( Fullconnect(2, 10, ReLU.function, ReLU.derivative, updater=GradientDescent(learning_rate)) )
        >>> n.layers.append( Fullconnect(10, 2, updater=GradientDescent(learning_rate)) )
        >>> n.activation = Identity()
        >>>
        >>> for epoch in range(0, 20):
        ...     loss = n.train( x = np.array([ [1, 2, 1, 2,  5, 6, 5, 6],
        ...                                    [5, 4, 4, 5,  1, 2, 2, 1]]).T,
        ...                target = np.array([ [1, 1, 1, 1,  0, 0, 0, 0],
        ...                                    [0, 0, 0, 0,  1, 1, 1, 1]]).T )
        ...     if epoch%5 == 0:
        ...         print 'epoch:%04d loss:%.2f'%(epoch, loss)
        epoch:0000 loss:52.82
        epoch:0005 loss:1.81
        epoch:0010 loss:1.26
        epoch:0015 loss:0.89
        >>>
        >>> y = n.predict( np.array( [[1, 6, 5], [5, 1, 4]] ).T )
        >>> print ['%.2f'%_ for _ in y[0]]
        ['1.19', '-0.00']
        >>>
        >>> # Auto-encoder
        >>> n = Network()
        >>> n.layers.append( Fullconnect( 2, 10, Tanh.function, Tanh.derivative, GradientDescent(learning_rate)) )
        >>> n.layers.append( Fullconnect(10, 10, Tanh.function, Tanh.derivative, GradientDescent(learning_rate)) )
        >>> n.layers.append( Fullconnect(10, 10, updater=NotUpdate()) )
        >>> n.layers.append( Fullconnect(10,  2, updater=NotUpdate()) )
        >>> n.activation = Identity()
        >>>
        >>> # for auto-encoder (weight share)
        >>> n.layers[2].W = n.layers[1].W.T
        >>> n.layers[3].W = n.layers[0].W.T
        >>>
        >>> x = np.array( [[1, 2, 1, 2,  5, 6, 5, 6,  5, 6, 5, 6],
        ...                [5, 4, 4, 5,  5, 4, 5, 4,  1, 2, 2, 1]] ).T
        >>>
        >>> for epoch in range(0, 301):
        ...     loss = n.train( x=x, target=x )
        ...     if epoch%100 == 0:
        ...         print 'epoch:%04d loss:%.2f'%(epoch, loss)
        epoch:0000 loss:98.38
        epoch:0100 loss:9.83
        epoch:0200 loss:1.79
        epoch:0300 loss:1.82
        '''

        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
