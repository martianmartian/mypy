import math
import numpy as np

class GradientDescent:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, param, gradient):
        param -= self.learning_rate * gradient
        return param

    def __test(self):
        '''
        >>> u = GradientDescent()
        >>> x = np.array( [[1, 2, 3], [2, 3, 4]], dtype=float )
        >>> grad = np.array( [[1, 2, 3], [2, 3, 4]], dtype=float )
        >>> x = u.update( x, grad )
        >>> [['%.2f'%_ for _ in v] for v in x]
        [['0.99', '1.98', '2.97'], ['1.98', '2.97', '3.96']]
        '''
        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
