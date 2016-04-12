import numpy as np

""" 
generates a 2D linearly separable dataset with n samples. 
The third element of the sample is the label
"""

def make_2DLinearSeparable_Dataset(n):
  xb = (np.random.rand(n)*2-1)/2-0.5
  yb = (np.random.rand(n)*2-1)/2+0.5
  xr = (np.random.rand(n)*2-1)/2+0.5
  yr = (np.random.rand(n)*2-1)/2-0.5
  inputs = []
  for i in range(len(xb)):
    inputs.append([xb[i],yb[i],1])
    inputs.append([xr[i],yr[i],-1])
  return inputs

# print np.random.rand(2)
# print make_2DLinearSeparable_Dataset(2)

""" 
hm..
"""
def make_OR_Dataset():
  training_data = [ 
    (np.array([0,0,1]), 0), 
    (np.array([0,1,1]), 1), 
    (np.array([1,0,1]), 1), 
    (np.array([1,1,1]), 1), 
  ]
  return training_data

# print make_OR_Dataset()

""" 
assembly last one
"""
def combine_X_y_Dataset():
  X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
  y = np.array([0, 1, 1, 0])
  ones = np.atleast_2d(np.ones(X.shape[0]))
  X = np.concatenate((ones.T, X), axis=1)
  training_data = []
  for i,val in enumerate(X):
    training_data.append((val,y[i]))
  return training_data



x = [1,2]
w = [3,4]
print np.dot(x,w)









