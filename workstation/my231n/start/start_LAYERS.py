

#    ) )
#    ( (
#  ........
# [|      |
#  \      /
#   `----'


import numpy as np
import matplotlib.pyplot as plt
import facilities.faci_start as faci

'''get data ready'''
from facilities.data_utils import get_CIFAR10_data_mini
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data_mini(newT=True)
D = X_train.shape[1]



'''get layer ready'''
from LAYERS.Affine import Affine

affine1 = Affine(K=3, D=D)
print affine1.forward_1(X_train)
# faci.printParams(affine1)

