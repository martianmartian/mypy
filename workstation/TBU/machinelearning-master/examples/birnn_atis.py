import cPickle as pkl
import numpy as np
import random
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from core.network import Network
from core.layers import Fullconnect, Recurrent, BiRecurrent
from core.activations import Softmax, Sigmoid
from core.nonlinears import Linear, ReLu, Tanh
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

    train, test, dicts = pkl.load( open('datas/atis.pkl', 'r') )
    index2words = {value:key for key, value in dicts['words2idx'].iteritems()}
    index2tables = {value:key for key, value in dicts['tables2idx'].iteritems()}
    index2labels = {value:key for key, value in dicts['labels2idx'].iteritems()}

    train_lex, train_ne, train_y = train
    test_lex, test_ne, test_y = test
    vocsize = len(dicts['words2idx']) + 1
    nclasses = len(dicts['labels2idx'])
    nsentences = len(train_lex)

    context_window_size = 7

    learning_rate = 0.01
    n = Network()
    n.layers.append( Fullconnect(vocsize, 100, Tanh.function, Tanh.derivative, updater=GradientDescent(learning_rate)) )
    n.layers.append( BiRecurrent(100, 100, Tanh.function, Tanh.derivative, updater=GradientDescent(learning_rate)) )
    n.layers.append( Fullconnect(100, nclasses, updater=GradientDescent(learning_rate)) )
    n.activation = Softmax()

    for epoch in range(0, 11):
        epoch_loss = 0
        for i in xrange(nsentences):
            cwords = contextwin(train_lex[i], context_window_size)
            words, labels = onehotvector(cwords, vocsize, train_y[i], nclasses)

            loss = n.train( words, labels ) / len(words) # sequence normalized loss
            epoch_loss += loss
            if i%1000 == 0:
                logging.info( 'epoch:%04d iter:%04d loss:%.2f'%(epoch, i, epoch_loss/(i+1)) )

        logging.info( 'epoch:%04d loss:%.2f'%(epoch, epoch_loss/nsentences) )

        for i in range(20):
            idx = random.randint(0, len(test_lex)-1)
            cwords = contextwin(test_lex[idx], context_window_size)
            words = onehotvector(cwords, vocsize)[0]
            labels = test_y[idx]
            _ = n.predict(words)
            y = [np.argmax(prediction) for prediction in _]
            #print _
            #print y

            print 'word:   ', ' '.join([index2words[_] for _ in test_lex[idx]])
            print 'label:  ', ' '.join([index2labels[_] for _ in labels])
            print 'predict:', ' '.join([index2labels[_] for _ in y])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-filename',         type=str, default='')
    args = parser.parse_args()

    if not args.log_filename:
        logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s %(levelname)s] %(message)s', stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s', filename=args.log_filename)

    main( args )

