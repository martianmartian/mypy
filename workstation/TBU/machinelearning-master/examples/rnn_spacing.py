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
    np.random.seed(0xC0FFEE)

    logging.info('load data start')
    train_lex, train_y = pkl.load( open('datas/kowiki_spacing_train.pkl', 'r') )
    words2idx = pkl.load( open('datas/kowiki_dict.pkl', 'r') )
    logging.info('load data done')

    index2words = {value:key for key, value in words2idx.iteritems()}

    vocsize = len(words2idx) + 1
    nclasses = 2
    nsentences = len(train_lex)

    context_window_size = args.window_size
    minibatch = args.minibatch
    learning_rate = args.learning_rate
    logging.info('vocsize:%d, nclasses:%d, window-size:%d, minibatch:%d, learning-rate:%.5f'%(vocsize, nclasses, context_window_size, minibatch, learning_rate))

    n = Network()
    n.layers.append( Fullconnect(vocsize, 256, Tanh.function, Tanh.derivative,  updater=GradientDescent(learning_rate)) )
    n.layers.append( Recurrent(256, 256, ReLU.function, ReLU.derivative, updater=GradientDescent(learning_rate)) )
    n.layers.append( Fullconnect(256, 256, ReLU.function, ReLU.derivative, updater=GradientDescent(learning_rate)) )
    n.layers.append( Fullconnect(256, nclasses, updater=GradientDescent(learning_rate)) )
    n.activation = Softmax(is_zero_pad=True)

    if os.path.isfile( args.params ):
        logging.info('load parameters from %s'%args.params)
        n.load_params( pkl.load(open(args.params, 'rb')) )

    logging.info('train start')
    for epoch in xrange(0, args.epoch):
        epoch_loss = 0
        epoch_error_rate = 0
        max_iterations = min(args.samples, nsentences) / minibatch
        for i in xrange( max_iterations ):
            max_size_of_sequence = 100
            idxs = [random.randint(0, nsentences-1) for _ in range(minibatch)]
            cwords = [contextwin(train_lex[idx][:max_size_of_sequence], context_window_size) for idx in idxs]
            words_labels = [onehotvector(cword, vocsize, train_y[idx][:max_size_of_sequence], nclasses) for idx, cword in zip(idxs, cwords)]

            words = [word for word, label in words_labels]
            labels = [label for word, label in words_labels]

            # zero padding for minibatch
            max_size_of_sequence = max( [_.shape[0] for _ in words] )
            for k, (word, label) in enumerate(zip(words, labels)):
                size_of_sequence = word.shape[0]
                words[k]  = np.pad(word,  ((0, max_size_of_sequence-size_of_sequence), (0, 0)), mode='constant')
                labels[k] = np.pad(label, ((0, max_size_of_sequence-size_of_sequence), (0, 0)), mode='constant')

            words  = np.swapaxes( np.array(words),  0, 1 )
            labels = np.swapaxes( np.array(labels), 0, 1 )

            loss = n.train( words, labels ) / (max_size_of_sequence * minibatch) # sequence normalized loss
            predictions = n.y
            error_rate = n.activation.error( predictions, labels ) / (max_size_of_sequence * minibatch)

            epoch_loss += loss
            epoch_error_rate += error_rate
            if i%10 == 0 and i != 0:
                logging.info('[%.4f%%] epoch:%04d iter:%04d loss:%.5f error-rate:%.5f'%((i+1)/float(max_iterations), epoch, i, epoch_loss/(i+1), epoch_error_rate/(i+1)))

        logging.info('epoch:%04d loss:%.5f, error-rate:%.5f'%(epoch, epoch_loss/max_iterations, epoch_error_rate/max_iterations))
        pkl.dump( n.dump_params(), open(args.params, 'wb') )
        logging.info('dump parameters at %s'%(args.params))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--window-size',          type=int,   default=1)
    parser.add_argument('--epoch',                type=int,   default=10)
    parser.add_argument('--minibatch',            type=int,   default=10)
    parser.add_argument('--learning-rate',        type=float, default=0.0001)
    parser.add_argument('-p', '--params',         type=str,   required=True)
    parser.add_argument('-n', '--samples',        type=int,   default=100000 )
    parser.add_argument('--log-filename',         type=str,   default='')
    args = parser.parse_args()

    if not args.log_filename:
        logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s %(levelname)s] %(message)s', stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s', filename=args.log_filename)

    main( args )

