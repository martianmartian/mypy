

#    ) )
#    ( (
#  ........
# [|      |
#  \      /
#   `----'
import numpy as np


# scores = np.random.random((2,5))
# print scores
# print scores[0,1]
# print scores[0][1]

# a = [0,1]
# b = [0,0]
# print scores[a,b]
# # print [a,b]
# # print scores[a][b]
# # print scores[[(0,0),(0,1),(0,1)]] nope

# values = np.random.random((100,5))
values = np.arange(0,500).reshape(100,5)
index_0 = range(values.shape[0])
index_1 = np.random.randint(low=0,high=values.shape[1],size=(values.shape[0],))
print values[index_0,index_1]