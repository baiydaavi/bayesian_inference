"""Marginalisation and 'contour plotting'.
"""

import numpy as np
from matplotlib import pyplot as plt



#Marginalised 2-D plot
def marginalised_2d(data_variable, bin1, bin2):
    """Takes in a (N x M)-sized array,  ignores the last M-2(M>=2) coordinates
    to produce a normalised histogram of the first 2 variables. 'bin1'
    and 'bin2' define the number of bins along each direction. 
    """
    plt.hist2d(x=data_variable[:, 0],y=data_variable[:, 1], bins=[bin1, bin2], density=1)
    plt.show()

#Fake Data
x=[]
for i in range(10000):
    col=[]

    col.append(np.random.normal(np.random.choice([2,20]), 0.5))
    col.append(np.random.normal(10, 0.5))
    col.append(np.random.normal(-2, 0.5))
    col.append(np.random.normal(-5, 0.5))
    x.append(col) 
x=np.array(x)

#Code execution
marginalised_2d(x, 40, 40)