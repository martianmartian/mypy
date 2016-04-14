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
    learning_rate = args.learning_rate
    minibatch = args.minibatch
    logging.info('vocsize:%d, nclasses:%d, window-size:%d, minibatch:%d, learning-rate:%.5f'%(vocsize, nclasses, context_window_size, minibatch, learning_rate))

    model_script = '''
def model(learning_rate=0.0001):
    n = Network()
    # word embedding layer
    n.layers.append( Fullconnect(573, 256,                       Tanh.function, Tanh.derivative,
        updater=GradientDescent(learning_rate)) )
    # recurrent layer
    n.layers.append( Recurrent(n.layers[-1].output_size, 256,    ReLU.function, ReLU.derivative,
        updater=GradientDescent(learning_rate)) )
    n.layers.append( Dropout(n.layers[-1].output_size, 256, 0.5, ReLU.function, ReLU.derivative,
        updater=GradientDescent(learning_rate)) )
    n.layers.append( Fullconnect(n.layers[-1].output_size, 127,
        updater=GradientDescent(learning_rate)) )
    n.activation = Softmax(is_zero_pad=True)
    return n
    '''
    exec( model_script )
    n = model( learning_rate )


    minimum_validation_error_rate = 1.0
    for epoch in xrange(args.epoch):
        for data in datas:
            for l, layer in enumerate(n.layers):
                if 'Dropout' == type( layer ).__name__:
                    n.layers[l].is_testing = data['name'] == 'test'

            epoch_loss = 0
            epoch_error_rate = 0
            max_iteration = data['size']/minibatch
            for i in xrange(max_iteration):
                if data['name'] == 'train':
                    idxs = [random.randint(0, data['size']-1) for _ in range(minibatch)]
                else:
                    idxs = [i*minibatch+k for k in range(minibatch)]
                cwords = [contextwin(data['x'][idx], context_window_size) for idx in idxs]
                words_labels = [onehotvector(cword, vocsize, data['y'][idx], nclasses) for idx, cword in zip(idxs, cwords)]

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

                if data['name'] == 'train':
                    loss = n.train( words, labels ) / (max_size_of_sequence * minibatch)
                    predictions = n.y
                else:
                    predictions = n.predict( words )
                    loss = n.activation.loss( predictions, labels ) / (max_size_of_sequence * minibatch)
                error_rate = n.activation.error( predictions, labels ) / (max_size_of_sequence * minibatch)

                epoch_loss += loss
                epoch_error_rate += error_rate
                if i%(1000/minibatch) == 0 and i != 0 and data['name'] == 'train':
                    logging.info( 'epoch:%04d iter:%04d loss:%.5f error-rate:%.5f'%(epoch, i, epoch_loss/(i+1), epoch_error_rate/(i+1)) )

            logging.info( '[%5s] epoch:%04d loss:%.5f error-rate:%.5f'%(data['name'], epoch, epoch_loss/max_iteration, epoch_error_rate/max_iteration) )
            if args.params and data['name'] == 'test' and minimum_validation_error_rate > epoch_error_rate/max_iteration:
                minimum_validation_error_rate = epoch_error_rate/max_iteration
                fname = args.params + '_min_error.pkl'
                pkl.dump( n.dump_params(), open(fname, 'wb') )
                logging.info('dump parameters at %s'%(fname))

    # prediction setup for evaluation
    for l, layer in enumerate(n.layers):
        if 'Dropout' == type( layer ).__name__:
            n.layers[l].is_testing = data['name'] == 'test'

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

        results['p'].append( [index2labels[_] for _ in y] )
        results['g'].append( [index2labels[_] for _ in labels] )
        results['w'].append( [index2words[_] for _ in x] )

    rv = conlleval(results['p'], results['g'], results['w'], 'atis_test_file.tmp')
    logging.info('evaluation result: %s'%(str(rv)))

    if args.params:
        fname = args.params + '_last.pkl'
        pkl.dump( n.dump_params(), open(fname, 'wb') )
        logging.info('dump parameters at %s'%(fname))

    '''
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
    '''



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--window-size',          type=int,   default=3)
    parser.add_argument('--epoch',                type=int,   default=30)
    parser.add_argument('--minibatch',            type=int,   default=10)
    parser.add_argument('--learning-rate',        type=float, default=0.003)
    parser.add_argument('-p', '--params',         type=str,   default='')
    parser.add_argument('--log-filename',         type=str,   default='')
    args = parser.parse_args()

    if not args.log_filename:
        logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s %(levelname)s] %(message)s', stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s', filename=args.log_filename)

    main( args )

