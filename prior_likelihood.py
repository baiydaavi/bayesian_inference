""" Defines the prior and likelihood functions.

This file contains the functions for the prior and likelihood
functions needed to calculate the posterior distribution over
the cosmological parameters.
"""

import numpy as np
import pandas as pd

# here is a stand-in for the mu_modell function we will eventually want. 
# We should only have to replace this line of code (kyle)
from temporary_functions import dummy_mu as mu_model

#made the default uniform, just for easier inputs when testing (kyle)
def prior(params, magnitude_mode='uniform'):

    """ This function calculates the prior.

    Parameter:
    1. params - This is a list of the 4 cosmological parameters
    omega_m, omega_lambda, H_0 and M.
    2. magnitude_mode - This parameter decides whether the prior
    on M (absolute magnitude) is 'uniform' or 'gaussian'.

    omega_m, omega_lambda and H_0 cannot take negative values therefore
    we choose the prior probability to be 0 when at least one of 
    the three parameters goes negative. For any non-negative values
    of omega_m, omega_lambda, and H_0, we choose a uniform prior probabiility.
    """

    # Prior is 0 if omega_m,omega_h, H_0 are negative.

    if any(i < 0 for i in params[0:-1]):
        return 0

    # Uniform prior on M

    if magnitude_mode == "uniform":
        return 1

    # Gaussian prior on corrected supernova absolute magnitude of
    # M =19.23 +/- 0.042.

    else:
        return np.exp(-0.5 * pow((params[3] - 19.23) / 0.042, 2))

# since the systematic error matrix is only going to be used here, I thought we could hardcore it into
# the function. I dont think this costs any overhead, but am not sure. It does make it integrate
# into the rest of the pipeline more smoothly though.
data_sys = pd.read_csv("sys_DS17f.txt", sep=" ")
data_sys.columns = ["sys_error"]
Sys_error_data = np.reshape(pd.Series.to_numpy(data_sys.sys_error), (40, 40))

# made default include_sys_error=False (kyle)
def likelihood(params, data_lcparam, include_sys_error=False, sys_error=Sys_error_data):

    """This function calculates the likelihood.

    Parameter:
    1. params - This is a list of the 4 cosmological parameters
    omega_m, omega_lambda, H_0 and M.
    2. mu_model - Input the mu_model function that analytically
    caculates mu. (Not needed if this is a part of the same file
    that defines the function mu_model.)
    2. data_lcparam - Importing the data file that contains the
    redshift, aparent magnitude and the statistical error data.
    3. sys_error - This is a 40x40 matrix that contains the 
    systematic error data.
    4. include_sys_error - This parameter decides whether the
    systematic error is included (value -> 'True') or excluded 
    (value -> 'False') from the covariance matrix calculation.
    """



    # Importing an array of size 40 that contains the apparent
    # magnitude data.

    app_mag = pd.Series.to_numpy(data_lcparam.mb)

    # Calculating the difference between the measured and
    # estimated apparent magnitude.

    diff_app_mag = app_mag - (mu_model(params, data_lcparam) - params[3])

    # Defining a 40x40 diagonal matrix whose diagonal entries
    # are the square of the corresponding statistical error.

    stat_error = np.diag(pow(pd.Series.to_numpy(data_lcparam.dmb), 2))

    # Only include the statistical error in the covariance
    # matrix calculation

    if include_sys_error == "False":
        inv_cov_matrix = np.linalg.inv(stat_error)

    # Include the systematic error as well in the covariance
    # matrix calculation.

    else:
        inv_cov_matrix = np.linalg.inv(stat_error + sys_error)

    return np.exp(-0.5 * (diff_app_mag @ inv_cov_matrix @ diff_app_mag))


# data_lcparam = pd.read_csv("lcparam_DS17f.txt", sep=" ")
# data_sys = pd.read_csv("sys_DS17f.txt", sep=" ")
# data_sys.columns = ["sys_error"]
# sys_error = np.reshape(pd.Series.to_numpy(data_sys.sys_error), (40, 40))
