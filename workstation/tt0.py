

#    ) )
#    ( (
#  ........
# [|      |
#  \      /
#   `----'

import scipy.misc
import matplotlib.pyplot

lena = scipy.misc.lena()
xmax = 50
ymax = 50
lena[0:xmax,0:ymax]=0
# xmax = lena.shape[0]
# ymax = lena.shape[1]
# lena[range(xmax), range(ymax)] = 0
# lena[range(xmax-1,-1,-1), range(ymax)] = 0

matplotlib.pyplot.imshow(lena)
matplotlib.pyplot.show()