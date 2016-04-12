import numpy as np
import matplotlib.pyplot as plt
from pylab import scatter, show, legend, xlabel, ylabel

data = np.loadtxt('./data/ex2data1.txt', delimiter=',')
''' becareful which data set used'''
x = np.array(data[:, 0:2])
y = np.array(data[:, 2])
alpha = 0.00045
m = x.shape[0]
ones = np.ones(m)
X = np.c_[ones,x]
J=[0]
theta = np.random.random(X.shape[1])
# print theta

for i in range(100):
  hypothesis = 1/(1+np.exp((-1)*np.dot(X,theta)))
  diff = hypothesis - y
  gradients = np.dot(diff,X)/m
  theta = theta - alpha*gradients
  y1 = (-1)*(np.dot(y,np.log(hypothesis)))
  y0 = np.dot(y-1,np.log(1-hypothesis + 0.00000001))
  Ji = (y1 + y0)/m
  J.append(Ji)

# print theta
# plt.plot(J,'o')
# plt.show()

# print 1/(1+np.exp((-1)*np.dot(X,theta)))>0.5

''' scatter plot data'''
pos = np.where(y == 1)
neg = np.where(y == 0)
plt.scatter(x[pos, 0], x[pos, 1], marker='o', c='b')
plt.scatter(x[neg, 0], x[neg, 1], marker='x', c='r')
plt.show()


=================== 2-D feature map ===============

import numpy as np
import matplotlib.pyplot as plt
from pylab import scatter, show, legend, xlabel, ylabel


def map_feature(x,degrees):
  X = np.ones(x.shape[0])
  if x.shape[1] == 2:    
      x1 = x[:,0]
      x2 = x[:,1]
  for degree in range(1,degrees):
      for i in range(0,degree+1):
        cal = (x1**(degree-i))*(x2**(i))
        X = np.c_[X,cal]
  return X

data = np.loadtxt('./data/ex2data2.txt', delimiter=',')
''' becareful which data set used'''
x = np.array(data[:, 0:2])
y = np.array(data[:, 2])
alpha = 0.045
m = x.shape[0]
J=[0]

X = map_feature(x,10)
theta = np.random.random(X.shape[1])
print theta

for i in range(1000):
  hypothesis = 1/(1+np.exp((-1)*np.dot(X,theta)))
  diff = hypothesis - y
  gradients = np.dot(diff,X)/m
  theta = theta - alpha*gradients
  y1 = (-1)*(np.dot(y,np.log(hypothesis+0.00000001)))
  y0 = np.dot(y-1,np.log(1-hypothesis + 0.00000001))
  Ji = (y1 + y0)/m
  J.append(Ji)

print theta
plt.plot(J,'o')
plt.show()


# first =  1/(1+np.exp((-1)*np.dot(X,theta))) > 0.5
# second = y > 0.5
# third = first == second
# print 1- third[np.where(third==False)].size/np.float(y.size)


==================   contour  =============
import numpy as np
import matplotlib.pyplot as plt
from pylab import scatter, show, legend, xlabel, ylabel


def map_feature(x,degrees):
  X = np.ones(x.shape[0])
  if x.shape[1] == 2:    
      x1 = x[:,0]
      x2 = x[:,1]
  for degree in range(1,degrees):
      for i in range(0,degree+1):
        cal = (x1**(degree-i))*(x2**(i))
        X = np.c_[X,cal]
  return X

data = np.loadtxt('./data/ex2data2.txt', delimiter=',')
x = np.array(data[:, 0:2])
y = np.array(data[:, 2])
alpha = 0.045
m = x.shape[0]
J=[0]
degrees = 10
X = map_feature(x,degrees)
theta = np.random.random(X.shape[1])

for i in range(10000):
  hypothesis = 1/(1+np.exp((-1)*np.dot(X,theta)))
  diff = hypothesis - y
  gradients = np.dot(diff,X)/m
  theta = theta - alpha*gradients
  y1 = (-1)*(np.dot(y,np.log(hypothesis+0.00000001)))
  y0 = np.dot(y-1,np.log(1-hypothesis + 0.00000001))
  Ji = (y1 + y0)/m
  J.append(Ji)



first =  1/(1+np.exp((-1)*np.dot(X,theta))) > 0.5
second = y > 0.5
# third = first == second
print 1- third[np.where(third==False)].size/np.float(y.size)


# # print theta
# # plt.plot(J,'o')
# # plt.show()

# u = np.linspace(-1, 1, 20)
# v = np.linspace(-1, 1, 20)
# z = np.zeros(shape=(len(u), len(v)))
# for i in range(len(u)):
#     for j in range(len(v)):
#         xx = np.c_[u[i],v[j]]        
#         XX = map_feature(xx,degrees)
#         z[j,i] = np.dot(XX,theta)
#         # z[j,i] = 1/(1+np.exp((-1)*np.dot(XX,theta)))

# plt.contour(u, v, z)
# # plt.contour(u, v, z, np.linspace(-1,1,10))

# pos = np.where(y == 1)
# print pos[0][-1]
# neg = np.where(y == 0)
# plt.scatter(x[pos, 0], x[pos, 1], marker='o', c='b')
# plt.scatter(x[neg, 0], x[neg, 1], marker='x', c='r')
# plt.show()
==================  regulate the 'overfitting'.. ==================
  # bigger lambda makes it very interesting....

import numpy as np
import matplotlib.pyplot as plt
from pylab import scatter, show, legend, xlabel, ylabel


def map_feature(x,degrees):
  X = np.ones(x.shape[0])
  if x.shape[1] == 2:    
      x1 = x[:,0]
      x2 = x[:,1]
  for degree in range(1,degrees):
      for i in range(0,degree+1):
        cal = (x1**(degree-i))*(x2**(i))
        X = np.c_[X,cal]
  return X

data = np.loadtxt('./data/ex2data2.txt', delimiter=',')
x = np.array(data[:, 0:2])
y = np.array(data[:, 2])
alpha = 0.045
m = x.shape[0]
lamb = 1
J=[0]
degrees = 6
X = map_feature(x,degrees)
theta = np.random.random(X.shape[1])

for i in range(10000):
  hypothesis = 1/(1+np.exp((-1)*np.dot(X,theta)))
  diff = hypothesis - y
  gradients = (np.dot(diff,X)+lamb*theta)/m
  gradients[0] = (np.dot(diff[0],X[0])/m)[0]
  theta = theta - alpha*gradients
  y1 = (-1)*(np.dot(y,np.log(hypothesis+0.00000001)))
  y0 = np.dot(y-1,np.log(1-hypothesis + 0.00000001))
  regulation = lamb/m*(np.sum(np.square(theta[1:len(theta)])))
  Ji = (y1 + y0)/m + regulation
  J.append(Ji)


     # '''============= accuracy estimation  ========='''
# first =  1/(1+np.exp((-1)*np.dot(X,theta))) > 0.5
# second = y > 0.5
# third = first == second
# print 1- third[np.where(third==False)].size/np.float(y.size)

#       # '''============= J plot  ========='''
# plt.plot(J,'o')
# plt.show()
      # '''============= contour plot  ========='''
u = np.linspace(-1, 1, 20)
v = np.linspace(-1, 1, 20)
z = np.zeros(shape=(len(u), len(v)))
for i in range(len(u)):
    for j in range(len(v)):
        xx = np.c_[u[i],v[j]]        
        XX = map_feature(xx,degrees)
        z[j,i] = np.dot(XX,theta)
        # z[j,i] = 1/(1+np.exp((-1)*np.dot(XX,theta)))

plt.contour(u, v, z)
# plt.contour(u, v, z, np.linspace(-1,1,10))
# plt.show()
      # '''============= scatter plot  ========='''
pos = np.where(y == 1)
# print pos[0][-1]
neg = np.where(y == 0)
plt.scatter(x[pos, 0], x[pos, 1], marker='o', c='b')
plt.scatter(x[neg, 0], x[neg, 1], marker='x', c='r')
plt.show()



========= last unfinished attempt ========
# import numpy as np
# import matplotlib.pyplot as plt
# from pylab import scatter, show, legend, xlabel, ylabel




# # data = np.loadtxt('./data/ex2data2.txt', delimiter=',')
# # x = np.array(data[:, 0:2])
# # y = np.array(data[:, 2])
# # alpha = 0.045
# # m = x.shape[0]
# # J=[0]
# # degrees = 8




# # pos = np.where(y == 1)
# # neg = np.where(y == 0)
# # plt.scatter(x[pos, 0], x[pos, 1], marker='o', c='b')
# # plt.scatter(x[neg, 0], x[neg, 1], marker='x', c='r')


# # plt.show()



import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')
# X, Y, Z = axes3d.get_test_data(0.05)

scale = 20
order = 4
X = np.arange(-scale*0.5, scale*0.5, 0.25)
Y = np.arange(-scale*0.5, scale*0.5, 0.25)
X, Y = np.meshgrid(X, Y)
# R = 5*(X)**2 + Y**2+X
# R = 5*(X)**2 + Y**2+X + X**3+Y**3
R = X**3*0.02+X**2+Y**2+(X*Y)
Z = (R)/(scale*order)-scale/4*2
# ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
cset = ax.contour(X, Y, Z, zdir='z', offset=-scale, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='x', offset=-scale, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='y', offset=scale, cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_xlim(-scale, scale)
ax.set_ylabel('Y')
ax.set_ylim(-scale, scale)
ax.set_zlabel('Z')
ax.set_zlim(-scale, scale)

plt.show()


============ what? ========
from numpy import linspace,meshgrid,exp
x = linspace(-50,50,131)
y = linspace(-50,50,101)
(X,Y) = meshgrid(x,y)
a = X**3*0.02+X**2+Y**2+(X*Y)

c = plt.contour(x,y,a,linspace(-1,100,11))
plt.clabel(c)
plt.grid()
plt.show()



==============================  X > 2-D
# X > 2-D; to be finished!!
X = np.array([[2,2],[2,3],[2,4]])
print "there are ",X.shape[1]," colums, they are: "
for xx in range(0,X.shape[1]):
    print X[:,xx]
x1 = X[:,0]
x2 = X[:,1]
# print x1,x2
degrees = 6
for degree in range(0,degrees):
    print degree,"====="
    for i in range(0,degree+1):
#         print degree-i,i
        print (x1**(degree-i))*(x2**(i))