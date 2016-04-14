import cPickle as pkl
import numpy as np
import random
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from core.network import Network
from core.layers import Fullconnect, Dropout, Recurrent
from core.activations import Softmax, Sigmoid
from core.nonlinears import Linear, ReLU, Tanh
from core.updaters import GradientDescent

from accuracy import conlleval

def contextwin(l, win):
    l = list(l)
    lpadded = win // 2 * [-1] + l + win // 2 * [-1]
    out = [lpadded[i:(i + win)] for i in range(len(l))]
    return out

def onehotvector(cwords, vocsize, y=[], nclasses=1):
    words = np.zeros( (len(cwords), vocsize) )
    labels = np.zeros( (len(cwords), nclasses) )
    for i, cword in enumerate( cwords ):
        for idx in cword:
            words[i][idx] = 1

    for i, label in enumerate( y ):
        labels[i][label] = 1
    return (words, labels)

def main(args):
    np.random.seed(0xC0FFEE)

    train, test, dicts = pkl.load( open('datas/atis.pkl', 'r') )
    index2words = {value:key for key, value in dicts['words2idx'].iteritems()}
    index2tables = {value:key for key, value in dicts['tables2idx'].iteritems()}
    index2labels = {value:key for key, value in dicts['labels2idx'].iteritems()}

    datas = [
            {'name':'train', 'x':train[0], 'y':train[2], 'size':len(train[0])},
            {'name':'test',  'x':test[0],  'y':test[2], 'size':len(test[0])},
            ]

    vocsize = len(dicts['words2idx']) + 1
    nclasses = len(dicts['labels2idx'])
    context_window_size = args.window_size

    n = Network()
    # word embedding layer
    n.layers.append( Fullconnect(vocsize, 256,                   Tanh.function, Tanh.derivative) )
    # recurrent layer
    n.layers.append( Recurrent(n.layers[-1].output_size, 256,    ReLU.function, ReLU.derivative) )
    n.layers.append( Dropout(n.layers[-1].output_size, 256, 0.5,  ReLU.function, ReLU.derivative) )
    n.layers.append( Fullconnect(n.layers[-1].output_size, nclasses) )
    n.activation = Softmax(is_zero_pad=True)

    if not os.path.isfile( args.params ):
        logging.error('not exist params: %s'%args.params)
        return

    fname = args.params
    n.load_params( pkl.load( open(fname, 'rb') ) )
    logging.info('load parameters at %s'%(fname))


    # prediction setup for evaluation
    for l, layer in enumerate(n.layers):
        if 'Dropout' == type( layer ).__name__:
            n.layers[l].is_testing = True

    data = datas[1]
    max_iteration = data['size']
    results = {'p':[], 'g':[], 'w':[]}
    for i in range(max_iteration):
        idx = i
        x = data['x'][idx]
        labels = data['y'][idx]

        cwords = contextwin(datas[1]['x'][idx], context_window_size)
        words = onehotvector(cwords, vocsize)[0]
        _ = n.predict(words)

        y = [np.argmax(prediction) for prediction in _]

        results['p'].append( [index2tables[_] for _ in y] )
        results['g'].append( [index2tables[_] for _ in labels] )
        results['w'].append( [index2words[_] for _ in x] )

    rv = conlleval(results['p'], results['g'], results['w'], 'atis_test_file.tmp')
    logging.info('evaluation result: %s'%(str(rv)))

    for i in range(20):
        idx = random.randint(0, datas[1]['size']-1)
        x = datas[1]['x'][idx]
        labels = datas[1]['y'][idx]

        cwords = contextwin(datas[1]['x'][idx], context_window_size)
        words = onehotvector(cwords, vocsize)[0]
        _ = n.predict(words)

        y = [np.argmax(prediction) for prediction in _]

        print 'word:   ', ' '.join([index2words[_] for _ in x])
        print 'table:  ', ' '.join([index2tables[_] for _ in labels])
        print 'label:  ', ' '.join([index2labels[_] for _ in labels])
        print 'predict:', ' '.join([index2labels[_] for _ in y])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--window-size',          type=int,   default=1)
    parser.add_argument('-p', '--params',         type=str,   default='')
    parser.add_argument('--log-filename',         type=str,   default='')
    args = parser.parse_args()

    if not args.log_filename:
        logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s %(levelname)s] %(message)s', stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s', filename=args.log_filename)

    main( args )

