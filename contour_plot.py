"""Testing for marginalisation and 'contour plotting'.
"""

import numpy as np
from matplotlib import pyplot as plt

def marginalised_histogram(data_variable, bin1, bin2, bin3=10, bin4=10):
    """Given a 2-parameter space(soon to be 4) data, this function 
    will plot the histogram of the first dimension by 'integrating'
    out the 2nd dimension.'data_variable' should be a (N x 2) numpy
    array. The number of bins for each variable are bin1, bin2, etc.
    Currently the marginalisation is over the first variable.
    """

    #Actual histogram evaluation
    #H is the count corresponding to the bin in edge. 'edges' are the
    #limits of the 
    H, edges= np.histogramdd(data_variable, bins=(bin1,bin2),density=1)
    y=np.sum(H,axis=1)
    plt.fill_between(edges[0],np.concatenate(([0],y)), step="pre")
    plt.show()

#Fake Data
x=[]
for i in range(1000):
    for j in range(2):
        col=[]
        col.append(np.random.normal(2,0.5))
        col.append(np.random.normal(10,0.5))
    x.append(col) 
x=np.array(x)

marginalised_histogram(x,10,5)