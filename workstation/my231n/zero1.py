

#    ) )
#    ( (
#  ........
# [|      |
#  \      /
#   `----'


import numpy as np


x = np.arange(10)[:,np.newaxis].T
print x.shape
indi = np.ones((4,1))
print indi
print np.dot(indi,x)
print "============"

w = np.ones((4,10))
indiw = np.random.choice([0,1],size=(4,1), p=[0.5,0.5])
print indiw
print indiw*w
print np.dot(indiw,w)