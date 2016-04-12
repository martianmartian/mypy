""" 
http://lcn.epfl.ch/tutorial/english/perceptron/html/intro.html
http://lcn.epfl.ch/tutorial/english/perceptron/html/instructions.html
http://lcn.epfl.ch/tutorial/english/perceptron/html/learning.html
http://glowingpython.blogspot.com/2011/10/perceptron.html
"""
from pylab import rand,plot,show,norm
import generateData
import perceptron

# print generateData.make2DLinearSeparableDataset(2)

trainset = generateData.make2DLinearSeparableDataset(30)
testset = generateData.make2DLinearSeparableDataset(20)
perceptron00 = perceptron.Perceptron()
print perceptron00.w
perceptron00.train(trainset)
print perceptron00.w

# print testset

for x in testset:
 r = perceptron00.response(x)
 if r != x[2]: # if the response is not correct
  print 'error'
 if r == 1:
  plot(x[0],x[1],'ob')  
 else:
  plot(x[0],x[1],'or')




# plot of the separation line.
# The separation line is orthogonal to w
n = norm(perceptron00.w)
ww = perceptron00.w/n
ww1 = [ww[1],-ww[0]]
ww2 = [-ww[1],ww[0]]
plot([ww1[0], ww2[0]],[ww1[1], ww2[1]],'--k')
show()



