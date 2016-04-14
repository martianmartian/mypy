# machinelearning [![Build Status](https://travis-ci.org/wbaek/machinelearning.svg?branch=master)](https://travis-ci.org/wbaek/machinelearning)

내용만 알면 간단하게 구현할 수 있는 machine learning에 대한 이론 정리 및 구현내용이다.

다음 코드는 각 모듈에서 필요한 코드만 발췌해 80여줄로 구현한 예이다.
[fullconnect multiclass classification source code](https://gist.github.com/wbaek/05201f13130fa005cb33)


각 모듈은 초기화 및 테스트코드를 제외한 10~20줄 내외로 정리되어있다.

현재는 코드만 있고 이론적인 내용은 추가해갈 예정이다.



## Features
* Bayes Classifier
  * [Naive Bayes Classifier] (core/bayes.py#L4)
  * [General Bayes Classifier] (core/bayes.py#L59)
* Network
  * [Feedforward Network](core/network.py)
  * [Autoencoder Network](README.md#auto-encoder)
  * [Recurrent Network](README.md#recurrent-network)
* Layer
  * [Fullconnect Layer](core/layers/fullconnect.py)
  * [Recurrent Layer](core/layers/recurrent.py)
  * [Dropout Layer](core/Layer/dropout.py)
  * ~~Convolution Layer~~
* Nonlinear Function
  * [Linear](core/nonlinears/linear.py)
  * [ReLU](core/nonlinears/relu.py)
  * [Tanh](core/nonlinears/tanh.py)
* Activation \w Negative Log Likelihood Loss
  * [Softmax \w cross-entropy error](core/activations/softmax.py)
  * [Sigmoid \w cross-entropy error](core/activations/sigmoid.py)
  * [Identity \w sum-of-squre error](core/activations/identity.py)
* Updater
  * ~~None~~
  * Vanila [Gradient Descent](core/updaters/gradient_descent.py)
  * ~~Momentum~~
  * ~~AdaGradient~~
* Initializer
  * [Xavier](core/layers/fullconnect.py#L17) (implemented in Layer init function)
  * ~~Kaiming Initializer~~

## Requirements
```
pip install -r requirements.txt
```


## Usage
### Multiclass Classification
* each input is assigned to one of K mutually exclusive classes.
```python
>>> from core.network import Network
>>> from core.layers import Fullconnect
>>> from core.nonlinears import ReLU
>>> from core.activations import Softmax
>>> from core.updaters import GradientDescent
>>> np.random.seed(0xC0FFEE)
>>> 
>>> n = Network()
>>> n.layers.append( Fullconnect(input_size=2, output_size=10, ReLU.function, ReLU.derivative, updater=GradientDescent(learning_rate=0.01)) )
>>> n.layers.append( Fullconnect(input_size=10, output_size=2, updater=GradientDescent(learning_rate=0.01)) )
>>> n.activation = Softmax()
>>> 
>>> for epoch in range(0, 20):
...     loss = n.train( x = np.array([ [1, 2, 1, 2,  5, 6, 5, 6],
...                                    [5, 4, 4, 5,  1, 2, 2, 1]]),
...                target = np.array([ [1, 1, 1, 1,  0, 0, 0, 0],
...                                    [0, 0, 0, 0,  1, 1, 1, 1]]) )
...     if epoch%5 == 0:
...         print 'epoch:%04d loss:%.2f'%(epoch, loss)
epoch:0000 loss:9.84
epoch:0005 loss:0.37
epoch:0010 loss:0.24
epoch:0015 loss:0.18
>>> 
>>> y = n.predict( np.array( [[1, 6, 3], [5, 1, 4]] ) )
>>> [_ for _ in np.argmax(y, 0)]
[0, 1, 0]
```


### Multiple-class Classification
```python
>>> from core.network import Network
>>> from core.layers import Fullconnect
>>> from core.nonlinears import ReLU
>>> from core.activations import Sigmoid
>>> from core.updaters import GradientDescent
>>> np.random.seed(0xC0FFEE)
>>> 
>>> n = Network()
>>> n.layers.append( Fullconnect(input_size=2, output_size=10, ReLU.function, ReLU.derivative, updater=GradientDescent(learning_rate=0.01)) )
>>> n.layers.append( Fullconnect(input_size=10, output_size=2, updater=GradientDescent(learning_rate=0.01)) )
>>> n.activation = Sigmoid()
>>> 
>>> for epoch in range(0, 20):
...     loss = n.train( x = np.array([ [1, 2, 1, 2,  5, 6, 5, 6,  5, 6, 5, 6],
...                                    [5, 4, 4, 5,  5, 4, 5, 4,  1, 2, 2, 1]]),
...                target = np.array([ [1, 1, 1, 1,  1, 1, 1, 1,  0, 0, 0, 0],
...                                    [0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 1, 1]]) )
...     if epoch%5 == 0:
...         print 'epoch:%04d loss:%.2f'%(epoch, loss)
epoch:0000 loss:17.45
epoch:0005 loss:9.05
epoch:0010 loss:5.83
epoch:0015 loss:3.97
>>> 
>>> y = n.predict( np.array( [[1, 6, 3, 5], [5, 1, 4, 5]] ) )
>>> [['%.2f'%_ for _ in v] for v in y]
[['0.96', '0.06', '0.95', '0.95'], ['0.13', '0.99', '0.56', '0.86']]
```

### Regression
```python
>>> from core.network import Network
>>> from core.layers import Fullconnect
>>> from core.nonlinears import ReLU
>>> from core.activations import Identity
>>> from core.updaters import GradientDescent
>>> np.random.seed(0xC0FFEE)
>>> 
>>> n = Network()
>>> n.layers.append( Fullconnect(input_size=2, output_size=10, ReLU.function, ReLU.derivative, updater=GradientDescent(learning_rate=0.01)) )
>>> n.layers.append( Fullconnect(input_size=10, output_size=2, updater=GradientDescent(learning_rate=0.01)) )
>>> n.activation = Identity()
>>> 
>>> for epoch in range(0, 20):
...     loss = n.train( x = np.array([ [1, 2, 1, 2,  5, 6, 5, 6,  5, 6, 5, 6],
...                                    [5, 4, 4, 5,  5, 4, 5, 4,  1, 2, 2, 1]]),
...                target = np.array([ [1, 1, 1, 1,  1, 1, 1, 1,  0, 0, 0, 0],
...                                    [0, 0, 0, 0,  1, 1, 1, 1,  1, 1, 1, 1]]) )
...     if epoch%5 == 0:
...         print 'epoch:%04d loss:%.2f'%(epoch, loss)
epoch:0000 loss:18.67
epoch:0005 loss:3.17
epoch:0010 loss:2.46
epoch:0015 loss:2.00
>>> 
>>> y = n.predict( np.array( [[1, 6, 3, 5], [5, 1, 4, 5]] ) )
>>> [['%.2f'%_ for _ in v] for v in y]
[['1.36', '0.43', '0.72', '0.54'], ['0.15', '0.69', '0.52', '0.63']]
```

### Auto-encoder
```python
>>> from core.network import Network
>>> from core.layers import Fullconnect
>>> from core.nonlinears import ReLU
>>> from core.activations import Identity
>>> from core.updaters import GradientDescent
>>> from core.updaters import NotUpdate
>>> np.random.seed(0xC0FFEE)
>>> 
>>> n = Network()
>>> n.layers.append( Fullconnect( 2, 10, Tanh.function, Tanh.derivative, GradientDescent(learning_rate=0.001)) )
>>> n.layers.append( Fullconnect(10, 10, Tanh.function, Tanh.derivative, GradientDescent(learning_rate=0.001)) )
>>> n.layers.append( Fullconnect(10, 10, updater=NotUpdate()) )
>>> n.layers.append( Fullconnect(10,  2, updater=NotUpdate()) )
>>> n.activation = Identity()
>>>
>>> # for auto-encoder (weight share)
>>> n.layers[2].W = n.layers[1].W.T
>>> n.layers[3].W = n.layers[0].W.T
>>>
>>> x = np.array( [[1, 2, 1, 2,  5, 6, 5, 6,  5, 6, 5, 6],
...                [5, 4, 4, 5,  5, 4, 5, 4,  1, 2, 2, 1]] )
>>>
>>> for epoch in range(0, 1001):
...     loss = n.train( x=x, target=x )
...     if epoch%100 == 0:
...         print 'epoch:%04d loss:%.2f'%(epoch, loss)
epoch:0000 loss:101.72
epoch:0100 loss:4.93
epoch:0200 loss:2.92
epoch:0300 loss:2.38
epoch:0400 loss:2.17
epoch:0500 loss:2.07
epoch:0600 loss:2.02
epoch:0700 loss:1.99
epoch:0800 loss:1.97
epoch:0900 loss:1.95
```

### Recurrent Network
```python
>>> import numpy as np
>>> from core.network import Network
>>> from core.layers import Fullconnect, Recurrent
>>> from core.activations import Softmax
>>> from core.nonlinears import Linear, ReLU, Tanh
>>> from core.updaters import GradientDescent
>>> 
>>> np.random.seed(0xC0FFEE)
>>> learning_rate = 0.01
>>> n = Network()
>>> n.layers.append( Recurrent(2, 10, Tanh.function, Tanh.derivative, updater=GradientDescent(learning_rate)) )
>>> n.layers.append( Fullconnect(10, 2, updater=GradientDescent(learning_rate)) )

>>> input  = np.array([[1, 2, 3, 4, 5, 4, 3, 2, 1, 0],
...                    [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]])
>>> target = np.array([[1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
...                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
>>> 
>>> for epoch in range(0, 1001):
...    loss = 0
...     n.init()
...     for x, t in zip(input.T, target.T):
...         loss += n.train( x.reshape(2, 1), t.reshape(2, 1) )
...     if epoch%10 == 0:
...         print 'epoch:%04d loss:%.2f'%(epoch, loss)
epoch:0000 loss:6.07
epoch:0010 loss:3.35
epoch:0020 loss:3.07
...
epoch:0980 loss:0.28
epoch:0990 loss:0.28
epoch:1000 loss:0.28
>>> n.init()
>>> for x, t in zip(input.T, target.T):
...     y = n.predict( x.reshape(2, 1) )
...     print 'x=', ','.join(['%.2f'%_ for _ in x]), 'y=', ','.join(['%.2f'%_ for _ in y])
x= 1.00,1.00 y= 1.00,0.00
x= 2.00,2.00 y= 1.00,0.00
x= 3.00,1.00 y= 1.00,0.00
x= 4.00,2.00 y= 0.99,0.01
x= 5.00,1.00 y= 0.97,0.03
x= 4.00,2.00 y= 0.18,0.82
x= 3.00,1.00 y= 0.97,0.03
x= 2.00,2.00 y= 0.97,0.03
x= 1.00,1.00 y= 1.00,0.00
x= 0.00,2.00 y= 1.00,0.00
```
