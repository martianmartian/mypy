# https://blog.dbrgn.ch/2013/3/26/perceptrons-in-python/
from random import choice 
from numpy import array, dot, random


unit_step = lambda x: 0 if x < 0 else 1

training_data = [ 
  (array([0,0,1]), 0), 
  (array([0,1,1]), 1), 
  (array([1,0,1]), 1), 
  (array([1,1,1]), 1), 
]

w = random.rand(3)

errors = []
eta = 0.2
n = 1000

for i in xrange(n): 
  x, expected = choice(training_data) 
  result = dot(w, x)
  error = expected - unit_step(result) 
  errors.append(error) 
  w += eta * error * x

# for x, _ in training_data: 
#   result = dot(x, w)
#   print("{}: {} -> {}".format(x[:2], result, unit_step(result)))

print("{}".format(w))


import matplotlib.pyplot as plt
plt.plot(errors, marker='o')
plt.title('Perceptron')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.show()