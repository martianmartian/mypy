import numpy as np

""" x2w: step one: make sure it's useing integers to get all w for it.

wFactor is amount of neurons at this layer, shouldn't be dependent on X. eventually load from file

eventually, X length should be arbitary, but Layer can only select <=len(X) amount of input
"""

"""there should be a third "adapt", "lower than certain threshold, unlearnes everything....."""
"""also, between habituateAT and enhanceAT, there should be an neutral zone."""


class Layer_w_tanh:
    def __init__(self,wFactor):
        self.response = np.tanh
        self.adapt={  
            'habituateAT':7,     # by nature
            'enhanceAT':2       # by nature
        }
        self.adaptFactor=0.1   # by nature
        self.threshold = 1 # changes, not up-bounded. such a weird thing to have.... changes based on enviroment... like ion concentraction
        self.wFactor=np.array(wFactor)
        self.wDict={}
        self.charge={}
        self.discharge={}
    def adaptW(self,i,Xi): 
        if Xi>=self.adapt['habituateAT']:
            self.wDict[i] = np.array(self.wDict[i]) - self.adaptFactor  #  need to refine it.. self.eta * error * x or something like that...
        elif Xi>=self.adapt['enhanceAT']:
            self.wDict[i] = np.array(self.wDict[i]) + self.adaptFactor
        else: #?? type change??
            print i, " weight not changed"
    def x2w(self,X):
        for i, wFactor in enumerate(self.wFactor):
            if X[i]>self.adapt['habituateAT']:
                X[i]=self.adapt['habituateAT']
            self.wDict[i]=[wFactor]*X[i].round(0)   # extend each list only, not changing value
    def chargeBY(self,X):
        for i, neuron in self.wDict.iteritems():
            self.charge[i] = np.array(neuron)*X[i]
            self.adaptW(i,X[i])
    def responseTO(self,X):
        for i, charge in self.charge.iteritems():
            # print i,charge
            if(charge[0]>=self.threshold):
                self.discharge[i]=self.response(charge)
            else:
                self.discharge[i]=0
        # print self.discharge
    def stimulateBY(self,X):
        self.x2w(X)
        # print "self.wDict",self.wDict
        self.chargeBY(X)
        self.responseTO(X)

wFactor=[0.9,0.9,0.7] # 0<=w<=1, load from file, none random.
group1 = Layer_w_tanh(wFactor)
X= np.array([8,1.5,1.5])
group1.stimulateBY(X)
# group1.stimulateBY(X)
# group1.stimulateBY(X)


# be careful the dependencies here.
# how to make sure X and wFactor same dimension?
# x squre, too big?...
# how to interpret output? relevance? or twitch strength?
