

""" 
generates a 2D linearly separable dataset with n samples. 
The third element of the sample is the label
"""
def make_2DLinearSeparable_Dataset(n):
  from pylab import rand
  xb = (rand(n)*2-1)/2-0.5
  yb = (rand(n)*2-1)/2+0.5
  xr = (rand(n)*2-1)/2+0.5
  yr = (rand(n)*2-1)/2-0.5
  inputs = []
  for i in range(len(xb)):
    inputs.append([xb[i],yb[i],1])
    inputs.append([xr[i],yr[i],-1])
  return inputs





def make_OR_Dataset():
  from numpy import array
  training_data = [ 
    (array([[0],[0],[1]]), array([[0]])), 
    (array([[0],[1],[1]]), array([[1]])), 
    (array([[1],[0],[1]]), array([[1]])), 
    (array([[1],[1],[1]]), array([[1]])), 
  ]
  return training_data

