""" Defines all the test functions.

This file contains all the test functions for the following
functions:
1. The function that calculates the apparent magnitude
2. The likelihood function
3. The metropolis-hastings function
4. The mcmc chain
"""

import numpy as np
from prior_likelihood import prior
from prior_likelihood import likelihood
import matplotlib.pyplot as plt
import pandas as pd

from theoretical_mag import calculate_apparent_mag as mag_model
from prior_likelihood import likelihood
from lambda_cdm_functions import lambda_cdm_likelihood
from lambda_cdm_functions import lambda_cdm_prior
from core_mcmc_functions import chain


def test_mag_func_omegaK_is_0():

    """ Apparent magnitude calculator test function.

    This is a test function that tests the apparent
    magnitude calculator function when Omega_K = 0.0.
    We use the following link to check if our answer
    is correct:
    http://www.astro.ucla.edu/~wright/CosmoCalc.html

    The apparent magnitude function outputs the apparent
    magnitude but in the website they calculate the 
    luminosity distance. Therefore, we calculate the 
    luminosity distance from the apparent magnitude output
    of our function to compare it with the one on the
    website.
    """

    # We choose the parameters such the Omega_M + Omega_lambda = 1.

    test_param = [0.286, 0.714, 69.6, -19.23]

    # Our apparent magnitude function takes the cosmological parameters
    # and the redshift data array as input. For the test, we input an
    # array containing one redshift data point.

    dist_modulus = mag_model(test_param, [3])[0]

    # Calculating the luminosity distance in units of 10^4 Mpc.

    lum_distance = round(10 ** ((dist_modulus - test_param[3]) / 5) * 10 / 10 ** 10, 2)

    # The expected output for the same parameter set in units of 10^4 Mpc.

    expected_output = 2.59

    assert (
        lum_distance == expected_output
    ), "apparent magnitude function does not work when OmegaK = 0.0"

    print("output of the magnitude function is correct")


def test_mag_func_omegaK_is_pos():

    """ Apparent magnitude calculator test function.

    This is a test function that tests the apparent
    magnitude calculator function when Omega_K > 0.0.
    We use the following link to check if our answer
    is correct:
    http://www.astro.ucla.edu/~wright/CosmoCalc.html

    The apparent magnitude function outputs the apparent
    magnitude but in the website they calculate the 
    luminosity distance. Therefore, we calculate the 
    luminosity distance from the apparent magnitude output
    of our function to compare it with the one on the
    website.
    """

    # We choose the parameters such the Omega_M + Omega_lambda < 1.

    test_param = [0.1, 0.714, 69.6, -19.23]

    # Our apparent magnitude function takes the cosmological parameters
    # and the redshift data array as input. For the test, we input an
    # array containing one redshift data point.

    dist_modulus = mag_model(test_param, [3])[0]

    # Calculating the luminosity distance in units of 10^4 Mpc.

    lum_distance = round(10 ** ((dist_modulus - test_param[3]) / 5) * 10 / 10 ** 10, 2)

    # The expected output for the same parameter set in units of 10^4 Mpc.

    expected_output = 3.30

    assert (
        lum_distance == expected_output
    ), "apparent magnitude function does not work when OmegaK > 0.0"

    print("output of the magnitude function is correct")


def test_mag_func_omegaK_is_neg():

    """ Apparent magnitude calculator test function.

    This is a test function that tests the apparent
    magnitude calculator function when Omega_K > 0.0.
    We use the following link to check if our answer
    is correct:
    http://www.astro.ucla.edu/~wright/CosmoCalc.html

    The apparent magnitude function outputs the apparent
    magnitude but in the website they calculate the 
    luminosity distance. Therefore, we calculate the 
    luminosity distance from the apparent magnitude output
    of our function to compare it with the one on the
    website.
    """

    # We choose the parameters such the Omega_M + Omega_lambda > 1.

    test_param = [0.5, 0.714, 69.6, -19.23]

    # Our apparent magnitude function takes the cosmological parameters
    # and the redshift data array as input. For the test, we input an
    # array containing one redshift data point.

    dist_modulus = mag_model(test_param, [3])[0]

    # Calculating the luminosity distance in units of 10^4 Mpc.

    lum_distance = round(10 ** ((dist_modulus - test_param[3]) / 5) * 10 / 10 ** 10, 2)

    # The expected output for the same parameter set in units of 10^4 Mpc.

    expected_output = 2.17

    assert (
        lum_distance == expected_output
    ), "apparent magnitude function does not work when OmegaK < 0.0"

    print("output of the magnitude function is correct")


def likelihood_test_fake_data():

    test_params = [0.3, 0.6, 75, -19.23]

    fake_data = pd.DataFrame(np.linspace(0.002, 2.0, 200), columns=["zcmb"])

    fake_data["dmb"] = np.random.uniform(0, 0.001, len(fake_data.zcmb))

    mean = mag_model(test_params, fake_data.zcmb)

    cov = np.diag(fake_data.dmb)

    fake_data["mb"] = np.random.multivariate_normal(mean, cov)

    omega_arr = np.arange(0.1, 1.0, 0.01)

    likelihood_mat = np.zeros((len(omega_arr), len(omega_arr)))

    for i, omega_m_item in enumerate(omega_arr):
        for j, omega_l_item in enumerate(omega_arr):
            likelihood_mat[i, j] = likelihood(
                [omega_m_item, omega_l_item, test_params[2], test_params[3]], fake_data
            )

    max_omega_m = omega_arr[int(np.argmax(likelihood_mat) / len(omega_arr))]
    max_omega_l = omega_arr[int(np.argmax(likelihood_mat) % len(omega_arr))]

    assert all(
        np.isclose(
            [max_omega_m, max_omega_l], [test_params[0], test_params[1]], atol=0.01
        )
    ), "likelihood function does not work"

    print("Likelihood function works well on the fake data.")


def mcmc_lambda_cdm_test():

    # import main data
    data_lcparam = pd.read_csv("lcparam_DS17f.txt", sep=" ")

    chn, rej, convergence_value = chain(
        data_lcparam,
        20000,
        1500,
        0.005,
        start_state=[0.8, 75, -19.23],
        variances=[0.1, 1.0, 0],
        prior_func=lambda_cdm_prior,
        likelihood_func=lambda_cdm_likelihood,
        prior_mode="uniform",
    )

    assert np.isclose(
        convergence_value[0], 0.284, atol=0.01
    ), "The chain doesn't converge to the right Omega_M value"

    print("mcmc chain works perfectly for the lambda cdm model")
