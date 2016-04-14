import cPickle as pkl
import numpy as np
import random
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from core.network import Network
from core.layers import Fullconnect, Recurrent
from core.activations import Softmax
from core.nonlinears import Linear, ReLU, Tanh
from core.updaters import GradientDescent

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
    logging.info('load data start')
    train_lex, train_y = pkl.load( open('datas/kowiki_spacing_train.pkl', 'r') )
    words2idx = pkl.load( open('datas/kowiki_dict.pkl', 'r') )
    logging.info('load data done')

    index2words = {value:key for key, value in words2idx.iteritems()}

    vocsize = len(words2idx) + 1
    nclasses = 2
    nsentences = len(train_lex)
    max_iter = min(args.samples, nsentences)
    logging.info('vocsize:%d, nclasses:%d, nsentences:%d, samples:%d, max_iter:%d'%(vocsize, nclasses, nsentences, args.samples, max_iter))

    context_window_size = args.window_size

    n = Network()
    n.layers.append( Fullconnect(vocsize, 256, Tanh.function) )
    n.layers.append( Recurrent(256, 256, ReLU.function) )
    n.layers.append( Fullconnect(256, 256, ReLU.function) )
    n.layers.append( Fullconnect(256, nclasses) )
    n.activation = Softmax(is_zero_pad=True)

    if not os.path.isfile( args.params ):
        logging.error('not exist parameter file: %s'%args.params)
        return

    n.load_params( pkl.load(open(args.params, 'rb')) )

    for i in xrange( max_iter ):
        cwords = contextwin(train_lex[i], context_window_size)
        words, labels = onehotvector(cwords, vocsize)

        y_list = [np.argmax(_) for _ in n.predict( words )]

        result_list = []
        for idx, y in zip(train_lex[i], y_list):
            if y == 1:
                result_list.append(' ')
            result_list.append( index2words[idx].encode('utf8') )
        print ''.join( result_list )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--window-size',          type=int, default=1)
    parser.add_argument('-p', '--params',         type=str, required=True)
    parser.add_argument('-n', '--samples',        type=int, default=1000 )
    parser.add_argument('--log-filename',         type=str, default='')
    args = parser.parse_args()

    if not args.log_filename:
        logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s %(levelname)s] %(message)s', stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s', filename=args.log_filename)

    main( args )

