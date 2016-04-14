import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import numpy as np
import cPickle as pkl
import logging

from core.network import Network
from core.layers import Fullconnect
from core.nonlinears import ReLu

def main(args):
    np.random.seed(0xC0FFEE)
    n = Network()
    n.layers.append( Fullconnect(2, 10, ReLu.function, ReLu.derivative) )
    n.layers.append( Fullconnect(10, 2) )

    x = np.array([[1, 2, 1, 2,  5, 6, 5, 6],
                  [5, 4, 4, 5,  1, 2, 2, 1]])
    t = np.array([[1, 1, 1, 1,  0, 0, 0, 0],
                  [0, 0, 0, 0,  1, 1, 1, 1]])

    for epoch in range(0, 20):
        loss = n.train( x, t )

    pkl.dump( n.dump_params().copy(), open(args.dump_params, 'wb') )
    logging.info('pickle dump done')


    nn = Network()
    nn.layers.append( Fullconnect(2, 10, ReLu.function, ReLu.derivative) )
    nn.layers.append( Fullconnect(10, 2) )


    nn.load_params( pkl.load( open('test.pkl', 'rb') ).copy() )
    logging.info('pickle load done')

    print 'before:', [['%.2f'%_ for _ in v] for v in n.predict( x )]
    print 'after: ', [['%.2f'%_ for _ in v] for v in nn.predict( x )]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump-params',          type=str, default='test-param.pkl')
    parser.add_argument('--log-filename',         type=str, default='')
    args = parser.parse_args()

    if not args.log_filename:
        logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s %(levelname)s] %(message)s', stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s', filename=args.log_filename)

    main( args )

