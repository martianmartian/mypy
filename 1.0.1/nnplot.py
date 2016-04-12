import matplotlib.pyplot as plt

class Plot_errors:
  def __init__(self,errors):
    self.errors = errors
    plt.plot(self.errors, marker='o')
    plt.title('Perceptron')
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.show()