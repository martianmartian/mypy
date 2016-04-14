import numpy as np

class Softmax(object):
    def __init__(self, is_zero_pad=False):
        self.eps = 1e-6
        self.is_zero_pad = is_zero_pad

    def forward(self, x):
        _ = np.exp(x)
        return np.divide(_, np.array(_.sum( axis=-1 ).T, ndmin=len(_.shape)).T)

    def backward(self, y, target):
        if self.is_zero_pad:
            input_size = y.shape[-1]
            batch_size = np.prod( y.shape )/input_size
            y_dot = y.reshape( (batch_size, input_size) )
            t_dot = target.reshape( (batch_size, input_size) )

            y_dot = np.dot(np.diag( [(1.0 if v>=0.5 else 0.0) for v in np.sum(t_dot, axis=-1)] ).T, y_dot)
            return (y_dot - t_dot).reshape( y.shape )
        else:
            return y - target

    def loss(self, y, target):
        return - np.sum( np.log(y+self.eps) * target )

    def error(self, y, target):
        input_size = y.shape[-1]
        batch_size = np.prod( y.shape )/input_size
        y_dot = y.reshape( (batch_size, input_size) )
        t_dot = target.reshape( (batch_size, input_size) )
        return batch_size - np.sum( [(1.0 if np.sum(t_dot[i]) <= 0.5 or t_dot[i][v] >= 0.5 else 0.0) for i, v in enumerate( np.argmax(y_dot, axis=-1) )] )

    def __test(self):
        '''
        >>> x = np.log( np.array([[1, 1], [12, 6], [3, 8]]) )
        >>> t = np.array([[1, 0], [0, 1], [1, 0]])
        >>> f = Softmax()
        >>> y = f.forward( x )
        >>> print [['%.2f'%_ for _ in v] for v in y]
        [['0.50', '0.50'], ['0.67', '0.33'], ['0.27', '0.73']]
        >>> d = f.backward(y, t)
        >>> print [['%.2f'%_ for _ in v] for v in d]
        [['-0.50', '0.50'], ['0.67', '-0.67'], ['-0.73', '0.73']]
        >>> l = f.loss(y, t)
        >>> print '%.2f'%l
        3.09
        >>> f = Softmax(is_zero_pad=True)
        >>> t = np.array([[1, 0], [0, 1], [0, 0]])
        >>> d = f.backward(y, t)
        >>> print [['%.2f'%_ for _ in v] for v in d]
        [['-0.50', '0.50'], ['0.67', '-0.67'], ['0.00', '0.00']]
        '''
        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
