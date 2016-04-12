
"""
start.py is procedural control.
    currently also contain upper stream "neurons". 
"""

import generateData
training_data = generateData.combine_X_y_Dataset()

import layer_w
layerwtanh_0 = layer_w.Layer_w_tanh(2+1)
layerwtanh_1 = layer_w.Layer_w_tanh(2+1)
layerwtanh_out = layer_w.Layer_w_tanh(1)

import feedbackSYS
feedback = feedbackSYS.Feedback_1(layerwtanh_0.response)  
""" initialize with desired response function """



from random import choice
errors = []
n = 1
for i in xrange(n):
  x, feedback.expected = choice(training_data)
  layerwtanh_0.output(x)


  # error = feedback.expected - layerw1.output(x)
  # layerw1.updateWeights(x,error)
  # errors.append(error) 


# from numpy import dot
# for x, _ in training_data: 
#   result = dot(x, layerw1.w)
#   print("{}: {} -> {}".format(x[:2], result, layerw1.output(x)))



# import nnplot
# nnplot.Plot_errors(errors)