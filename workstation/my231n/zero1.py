

#    ) )
#    ( (
#  ........
# [|      |
#  \      /
#   `----'


import numpy as np

a = np.arange(10)
# a = a[:,np.newaxis]
a = a[np.newaxis,:]
print a.shape