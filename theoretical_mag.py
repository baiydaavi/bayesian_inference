'''
Calculate distance modulus and apparent magnitude of SN Ia based on their 
observed redshift, assumed absolute magnitude,  as well as a LambdaCDM-K
cosmology model assuming values for omegaM, omegaLambda, omegaK, and H0. 

@authors: aavinash@ucdavis.edu, kjray@ucdavis.edu, pjgandhi@ucdavis.edu,
ssreedhar@ucdavis.edu
'''

import math
import numpy as np
from scipy import integrate


def calculate_apparent_mag(params, SNdata):
    '''
    Parameters
    ----------
    Computes apparent magnitudes for an array of observed SN by using their 
    redshift data along with assumed cosmology parameters to compute a 
    luminosity distance for each SN, after which it is converted into a 
    distance modulus and combined with an input absolute magnitude to finally
    ouput apparent magnitudes.

    Note that for calculating luminosity distance we use the following
    relations based on curvature:
    For k = 0: dL = (c(1+z)/H0)*(int^z_0 dz/E(z))
    For k > 0: dL = (c(1+z)/H0)*(1/sqrt(abs(OmegaK))) \
                    *sin(sqrt(abs(OmegaK))*(int^z_0 dz/E(z)))
    For k < 0: dL = (c(1+z)/H0)*(1/sqrt(abs(OmegaK))) \
                    *sinh(sqrt(abs(OmegaK))*(int^z_0 dz/E(z)))
    Here E(z) = sqrt(OmegaM(1+z)^3 + OmegaLambda + OmegaK(1+z)^2)

    Also note that in order to get the final dL in units of [pc] for use
    in calculating the distance modulus, we input c in units of [km/s] 
    and H0 in units of [km/s/pc]. The value of c is hard-coded in while
    H0 is multiplied by 10^-6 to convert from [km/s/Mpc] to [km/s/pc].

    params: array of 4 input parameters based on our assumed LambdaCDM-K
            cosmology model (in order)
            OmegaM: energy density of matter [dimensionless]
            OmegaLambda: energy density of dark energy [dimensionless]
            H0: present-day value of Hubble parameter [needs to be in km/s/Mpc]
            M: fiducial SN Ia absolute magnitude [dimensionless]
    SNdata: 2D data table consisting of the following values for each 
            SN Ia (in order)
            mb: apparent magnitude in B band [dimensionless]
            dmb: error on apparent magnitude [dimensionless]
            zhel: redshift [dimensionless]
    
    Returns
    -------
    apparent_mags: array of calculated apparent magnitude values for 
                   each SN Ia based on our assumed cosmology
    '''

    c = 3.0*(10**5)                       # speed of light in km/s
    apparent_mags = [0.0]*len(SNdata)     # to store calculated m values for each SN

    OmegaK = 1 - params[0] - params[1]    # based on sum(Omega_i) = 1
    
    # looping over each SN in the dataset
    for i in range(0, len(SNdata['col2'])):
        
        f = lambda x: ((params[0]*((1+x)**3)) + (params[1]) + \
                      (OmegaK*((1+x)**2)))**-0.5
        eta, etaerr = integrate.quad(f, 0.0, SNdata['col2'][i])
        
        # note here that dL ends up being in units of pc
        if OmegaK == 0.0:
            dL = (c*(1+SNdata['col2'][i])/(params[2]*(10**-6))) * eta  
        elif OmegaK > 0.0:
            dL = (c*(1+SNdata['col2'][i])/(params[2]*(10**-6))) * (1/math.sqrt(abs(OmegaK))) \
                 * math.sin(math.sqrt(abs(OmegaK)) * eta)
        elif OmegaK < 0.0:
            dL = (c*(1+SNdata['col2'][i])/(params[2]*(10**-6))) *(1/math.sqrt(abs(OmegaK))) \
                 * math.sinh(math.sqrt(abs(OmegaK)) * eta)
        
        apparent_mags[i] = 5*math.log10(dL/10.0) + params[3]
    
    return apparent_mags
