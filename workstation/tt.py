

#    ) )
#    ( (
#  ........
# [|      |
#  \      /
#   `----'


import numpy as np

import matplotlib.pyplot as plt
from cs231n.classifier_trainer import ClassifierTrainer
from cs231n.gradient_check import eval_numerical_gradient
from cs231n.classifiers.convnet import *


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-18, np.abs(x) + np.abs(y))))

from cs231n.data_utils import load_CIFAR10
from cs231n.data_utils import load_CIFAR10_mini