'''This script runs a standard unit test on the function
used to calculate theoretical apparent magnitude values 
from observed SN redshifts, based on our assumed cosmology.
The test reproduces figure 11 from Scolnic et al. 2018
as a check on the function's accuracy.'''

import matplotlib.pyplot as plt

def test_apparent_mag(params, SNdata):
    '''                                                                                   
    Testing with default parameters of:
    OmegaM = 0.3, OmegaLambda = 0.7, H0 = 70km/s/Mpc, M = -19.3
    '''
    
    params = [0.3, 0.7, 70, -19.3]
    
    m = calculate_apparent_mag(params, SNdata)
    
    mu = []
    for i in range(0, len(m)):
        mu.append(m[i] - params[3])
    
    plt.scatter(SNdata['zhel'], mu)
    plt.xlabel('Redshift [z]', fontsize=12)
    plt.ylabel('Distance Modulus [$\\mu$]', fontsize=12)
    plt.xscale('log')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.show()
