import numpy as np

if __name__=='__main__':
    f = np.array([123, 456, 789]) # example with 3 classes and each having large scores
    p = np.exp(f) / np.sum(np.exp(f)) # Bad: Numeric problem, potential blowup

    # instead: first shift the values of f so that the highest number is 0:
    f -= np.max(f) # f becomes [-666, -333, 0]
    p = np.exp(f) / np.sum(np.exp(f)) # safe to do, gives the correct answer

