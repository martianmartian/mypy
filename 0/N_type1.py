currently, the NN neuron vs. synapses is alreayd scaling 
very well, if not for storage...
so the bottle neck is trainning speed........
the algorithm sucks. need more strategies to improve. 


considering the size of fiels, is it pssible... weights is limited to 
only a few digits after decimal..

import numpy as np 

# N_type1: only receive positive inputs
# so when number is converted, it only goes to positive side
N_typ1 =np.array([[1,-4,1,1,1,0,1,1,0],
                  [1,-4,1,1,1,0,1,1,0],
                  [1,-4,1,1,1,0,1,1,0],
                  [1,-4,1,1,1,0,1,1,0],
                  [1,-4,1,1,1,0,1,1,0]])
# these stores teh weights of input... really?...
# infor stored in neuron can only be integers to save space. 

def sim_type1(i):
    # single_input_matrix
    # convert a integer into binray of 10 digit
    # 4 ==> 000010000
    lis = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    lis[i] = 1
    return lis

# def sum_matrix(s):
def association_matrix(**s):
  # allows arbitary amount of input.
  # con-catenate all the input matrix into 01 binary array, 
  # identify ... where to find injection points.?
  # simulate sum of input at soma.. (distance..? nay)
  # this function needs to know, which type of inputs are involved..
  # ++, +++, ++-, --+, etc
  # to simplify algo, can sign be integrated into signal itself? 
  # lis = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, additional digit here]
    lis = [0] * (13 + len(settings.strategies))
    lis[s] = 1
    return lis


