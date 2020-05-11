import numpy as np
import pandas as pd
import math
from scipy import integrate


def lambda_cdm_mag(params, redshift_data):
    """
    Computes apparent magnitudes for an array of observed SN by using their 
    redshift data along with assumed cosmology parameters to compute a 
    luminosity distance for each SN, after which it is converted into a 
    distance modulus and combined with an input absolute magnitude to finally
    ouput apparent magnitudes.
    
    All calculations of luminosity distances and distance moduli done here
    are based on equations (9) and (10) from the Scolnic et al. 2018 paper. 

    Note that for calculating luminosity distance we use the following
    relations based on curvature. These relations have been derived from
    equations (9) and (10) from Scolnic et al. 2018, by taking the appropriate
    limits for positive, negative, and zero curvature cases:

    dL = (c(1+z)/H0)*(int^z_0 dz/E(z))

    Here E(z) = sqrt(OmegaM(1+z)^3 + OmegaLambda + OmegaK(1+z)^2)

    Also note that in order to get the final dL in units of [pc] for use
    in calculating the distance modulus, we input c in units of [km/s] 
    and H0 in units of [km/s/pc]. The value of c is hard-coded in while
    H0 is multiplied by 10^-6 to convert from [km/s/Mpc] to [km/s/pc].
    
    Parameters
    ----------
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
    """

    c = 3.0 * (10 ** 5)  # speed of light in km/s

    lambda_cdm_mag = [0.0] * len(
        redshift_data
    )  # to store calculated m values for each SN

    # looping over each SN in the dataset

    for i in range(0, len(redshift_data)):

        f = lambda x: ((params[0] * ((1 + x) ** 3)) + (1 - params[0])) ** -0.5

        eta, _ = integrate.quad(f, 0.0, redshift_data[i])

        # note here that dL ends up being in units of pc

        dL = (c * (1 + redshift_data[i]) / (params[1] * (10 ** -6))) * eta

        lambda_cdm_mag[i] = 5 * math.log10(dL / 10.0) + params[2]

    return lambda_cdm_mag


# made the default uniform, just for easier inputs when testing (kyle)
def lambda_cdm_log_prior(params, magnitude_mode="uniform"):

    """ This function calculates the prior.

    Parameter:
    1. params - This is a list of the 4 cosmological parameters
    omega_m, omega_lambda, H_0 and M.
    2. magnitude_mode - This parameter decides whether the prior
    on M (absolute magnitude) is 'uniform' or 'gaussian'.

    omega_m, omega_lambda and H_0 cannot take negative values therefore
    we choose the prior probability to be 0 when at least one of 
    the three parameters goes negative( or when omega_m and/or 
    omega_lambda gets higher than 2.5). For any other values of 
    omega_m, omega_lambda, and H_0, we choose a uniform prior 
    probabiility.
    """

    # Prior is 0 if omega_m and/or omega_lambda and/or H_0 are negative.

    if any(i <= 0 for i in params[0:-1]):
        return "forbidden"

    # Prior is 0 if omega_m and/or omega_lambda are greater than 2.5.

    if params[0] > 1.0:
        return "forbidden"

    if magnitude_mode == "uniform":
        return 0

    # Gaussian prior on corrected supernova absolute magnitude of
    # M =19.23 +/- 0.042.

    else:
        return -0.5 * pow((params[3] + 19.23) / 0.042, 2)


def lambda_cdm_log_likelihood(params, data_lcparam, sys_error=None):

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
    systematic error data. The default value for this argument is
    None. This means that if this argument isn't passed into this
    function then the systematic error isn't included in the
    covariance matrix calculation.
    """

    # Importing an array of size 40 that contains the apparent
    # magnitude data.

    app_mag = pd.Series.to_numpy(data_lcparam.mb)

    # Calculating the difference between the measured (app_mag)
    # and estimated apparent magnitude (mag_model).

    diff_app_mag = app_mag - lambda_cdm_mag(params, data_lcparam.zcmb)

    # Defining a 40x40 diagonal matrix whose diagonal entries
    # are the square of the corresponding statistical error.

    stat_error = np.diag(pow(pd.Series.to_numpy(data_lcparam.dmb), 2))

    # Only include the statistical error in the covariance
    # matrix calculation

    if sys_error is None:
        inv_cov_matrix = np.linalg.inv(stat_error)

    # Include the systematic error as well in the covariance
    # matrix calculation.

    else:
        inv_cov_matrix = np.linalg.inv(stat_error + sys_error)

    return -0.5 * (diff_app_mag @ inv_cov_matrix @ diff_app_mag)
