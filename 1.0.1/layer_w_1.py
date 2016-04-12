
from numpy import dot, random



class Layer_w_1:
  """ w currently is the property of current layer """
  def __init__(self):
    self.w = random.rand(3)
    self.eta = 0.2
    self.response = lambda x: 0 if x < 0 else 1

  # step one:
  def stimulate(self,x):
    return dot(self.w, x) 

  def output(self,x):
    charge = self.stimulate(x)
    return self.response(charge)

  def updateWeights(self,x,error):
    self.w += self.eta * error * x




# layerw1 = Layer_w_1()
# print layerw1.response(1)