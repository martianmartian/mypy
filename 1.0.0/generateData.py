from pylab import rand,plot,show,norm

def make2DLinearSeparableDataset(n):
 """ 
  generates a 2D linearly separable dataset with n samples. 
  The third element of the sample is the label
 """
 xb = (rand(n)*2-1)/2-0.5
 yb = (rand(n)*2-1)/2+0.5
 xr = (rand(n)*2-1)/2+0.5
 yr = (rand(n)*2-1)/2-0.5
 inputs = []
 for i in range(len(xb)):
  inputs.append([xb[i],yb[i],1])
  inputs.append([xr[i],yr[i],-1])
 return inputs

# print make2DLinearSeparableDataset(1)
# print rand(3)
# print (rand(2)*2-1)/2+0.5

"""
n = 1
inputs = 
[ 
  [-0.71661326833319861, 0.85034444106779805, 1], 
  [0.75288295528685367, -0.46246884883481942, -1]
]

n = 2
inputs = 
[ 
  [-0.71661326833319861, 0.85034444106779805, 1], 
  [0.75288295528685367, -0.46246884883481942, -1], 
  [-0.37045623856992915, 0.44038416103603961, 1], 
  [0.045645372736330403, -0.14326502520712681, -1]
]
"""

"""
          print rand(2)
[ 0.31496981  0.1014561 ]
[ 0.00811603  0.8052696 ]
          print rand(3)
[ 0.44083205  0.78917833  0.73917123]

     print (rand(2)*2-1)/2-0.5  -1to0 xb yr
[-0.6422542  -0.29730448]
[-0.90137528 -0.24380716]

     print (rand(2)*2-1)/2+0.5  0to1
"""
# print rand(2)*2-1
# print rand(2)