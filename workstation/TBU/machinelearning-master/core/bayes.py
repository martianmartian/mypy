import numpy as np
from scipy.stats import multivariate_normal

class NaivBayes(object):
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.priors = [0] * num_classes
        self.means  = [None] * num_classes
        self.covs   = [None] * num_classes

    def __str__(self):
        return 'num_classes:%d, priors:%s, means:%s, covs:%s'%(
                self.num_classes, str(self.priors), str(self.means), str(self.covs) )

    def predict(self, x):
        _ = np.array( [self.priors[i] * multivariate_normal.pdf(x, mean=m, cov=c, allow_singular=True)
            for i, (m, c) in enumerate(zip(self.means, self.covs))] )
        norm = np.sum(_, axis=0)
        _ = np.divide(_, norm)
        return _.T

    def train(self, x, target):
        self.priors = [sum([1.0 for t in target if t==k])/len(target) for k in range(self.num_classes)]
        self.means = [ np.mean( [v for v, t in zip(x, target) if t==k], axis=0 ) for k in range(self.num_classes) ]
        self.covs = [ np.diag( np.var( [v for v, t in zip(x, target) if t==k], axis=0 ) ) for k in range(self.num_classes) ]

    def dump_params(self):
        odict = {}
        odict['num_classes'] = self.num_classes
        odict['priors'] = self.priors
        odict['means']  = self.means
        odict['covs']   = self.covs
        return odict

    def load_params(self, idict):
        self.num_classes = idict['num_classes']
        self.priors = idict['priors']
        self.means  = idict['means']
        self.covs   = idict['covs']

    def __test(self):
        '''
        >>> # Naive Vayes classification
        >>> c = NaivBayes(2)
        >>> c.train( x = np.array([ [1, 2,  5, 7, 7, 5, 5, 7],
        ...                         [5, 4,  1, 2, 2, 1, 1, 2]]).T,
        ...     target = np.array(  [0, 0,  1, 1, 1, 1, 1, 1] ) )
        >>> str(c)
        'num_classes:2, priors:[0.25, 0.75], means:[array([ 1.5,  4.5]), array([ 6. ,  1.5])], covs:[array([[ 0.25,  0.  ],\\n       [ 0.  ,  0.25]]), array([[ 1.  ,  0.  ],\\n       [ 0.  ,  0.25]])]'
        >>> y = c.predict( np.array( [[1, 6, 3],
        ...                           [5, 1, 4]] ).T )
        >>> print ['%.2f'%_ for _ in y[0]]
        ['1.00', '0.00']
        >>> [_ for _ in np.argmax(y, -1)]
        [0, 1, 0]
        '''
        pass

class GeneralBayes(NaivBayes):
    def train(self, x, target):
        self.priors = [sum([1.0 for t in target if t==k])/len(target) for k in range(self.num_classes)]
        self.means = [ np.mean( [v for v, t in zip(x, target) if t==k], axis=0 ) for k in range(self.num_classes) ]
        self.covs = [ np.cov( np.array([v for v, t in zip(x, target) if t==k]).T ) for k in range(self.num_classes) ]

    def __test(self):
        '''
        >>> # Naive Vayes classification
        >>> c = GeneralBayes(2)
        >>> c.train( x = np.array([ [1, 2,  5, 7, 6, 6, 5, 7],
        ...                         [5, 4,  1, 2, 2, 1, 1, 2]]).T,
        ...     target = np.array(  [0, 0,  1, 1, 1, 1, 1, 1] ) )
        >>> str(c)
        'num_classes:2, priors:[0.25, 0.75], means:[array([ 1.5,  4.5]), array([ 6. ,  1.5])], covs:[array([[ 0.5, -0.5],\\n       [-0.5,  0.5]]), array([[ 0.8,  0.4],\\n       [ 0.4,  0.3]])]'
        >>> y = c.predict( np.array( [[1, 6, 3],
        ...                           [5, 1, 4]] ).T )
        >>> print ['%.2f'%_ for _ in y[0]]
        ['1.00', '0.00']
        >>> [_ for _ in np.argmax(y, -1)]
        [0, 1, 0]
        '''
        pass


if __name__ == "__main__":
    import doctest
    doctest.testmod()
