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
from core_mcmc_functions import metropolis


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

    chn, _, convergence_value = chain(
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


Data_lcparam = pd.read_csv("lcparam_DS17f.txt", sep=" ")


def likelihood_test(
    resolution,
    p1_min=0.1,
    p1_max=1.2,
    p2_min=0.5,
    p2_max=1.15,
    data=Data_lcparam,
    p1_slice=0.52,
    p2_slice=0.82,
    two_d=True,
    save=False,
):
    """
    this function does a brute force likelihood sweep of the omegas setting M=74 and H0=-19.23. It might not be a
    'unit test' (in that there is no simple assert statement), but we found it to be invaluable when diagnosing bugs

    params
    -----
    resolution : int
        number of points to sample on each axis
    p1_min, p1_max, p2_min, p2_max : numbers in the domain of the respective omega
        the min/max value of the omegas we want to test
    data: pandas data frame
        this is the data
    p1_slice(p2_slice) : numbers in the domain of omega_m(omega_lambda)
        this is the value we hold constant of each parameter when making the 1d plots
    two_d : True/False
        if false, we only do 1D plots. will be much faster
    save: True/False
        True for save, False for no save

    returns
    ------
    shows and/or saves plots. no returns
    """
    p1 = np.linspace(p1_min, p1_max, resolution)
    p2 = np.linspace(p2_min, p2_max, resolution)
    p1_lik = np.zeros(resolution)
    p2_lik = np.zeros(resolution)

    names = dict([(0, "$\\Omega_m$"), (1, "$\\Omega_\\Lambda$")])

    for i, p1_item in enumerate(p1):
        p1_lik[i] = likelihood([p1_item, p2_slice, 74, -19.23], data)
    for j, p2_item in enumerate(p2):
        p2_lik[j] = likelihood([p1_slice, p2_item, 74, -19.23], data)

    plt.rc("axes", titlesize=12)
    plt.rc("axes", labelsize=18)
    plt.rc("figure", titlesize=20)

    if two_d:
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    else:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    for item in ax:
        item.set_xlabel(names[0])
        item.set_ylabel("log(likelihood)")

    ax[0].plot(p1, p1_lik)
    ax[0].set_title(
        "log likelihood of " + names[0] + " when " + names[1] + " ={}".format(p2_slice)
    )
    ax[1].plot(p2, p2_lik)
    ax[1].set_title(
        "log likelihood of " + names[1] + " when " + names[0] + "={}".format(p1_slice)
    )

    if two_d:
        x, y = np.meshgrid(p1, p2)
        z = np.zeros((resolution, resolution))
        for i, p1_item in enumerate(p1):
            for j, p2_item in enumerate(p2):
                z[i, j] = likelihood([p1_item, p2_item, 74, -19.23], data)
                print("{:2.1%} done".format(i / resolution), end="\r")

        c1 = np.max(z) - 2.3 / 2
        c2 = np.max(z) - 6.17 / 2
        contours = [c2, c1, np.max(z)]
        c = ax[2].contour(x, y, z, contours)
        fig.colorbar(c, ax=ax[2])
        ax[2].set_title("log_likelihood when H0=74 and M=-19.23 are set")
    plt.show()
    if save:
        plt.savefig("likelihood_test{}".format(resolution))


def chain_test(data=Data_lcparam):
    """
    this runs the chain algorithm, setting 3 paramters fixed and makes sure that
    the third parameter converges to the value that is the maximum likelihood value
    when the others are fixed. returns an assert error if it fails
    """
    chn, _1, _2 = chain(
        data,
        1000,
        400,
        0.01,
        start_state=[0.2, 0.82, 74, -19.23],
        variances=[0.05, 0, 0, 0],
        prior_mode="uniform",
    )
    mu = np.mean(chn[200:, 0])
    assert mu > 0.36 and mu < 0.4, "chain failed to head towards max likelihood "
    print("no problems detected")


def metropolis_test():
    """
    Unit test for the metropolis part of the codebase. First, we create simple likelihood (shifted gaussian) and prior(uniform)
    Then, we make a data set where we know the answer by setting the numpy seed=0.
    Using this data set, we first verify that metropolis returns True when the proposed jump is to a higher likelihood region
    Then, we varify that jumps to lower likelihood region have approximately correct acceptance proportion over 10,000 trials
    The correct proportion was calibrated to be ~45.5%, but it allows a small region around that, because we are taking finite samples.
    """

    def log_likelihood(data, param):
        return -0.5 * np.sum((data - param) ** 2)

    def uniform_log_prior(params, magnitude_mode="uniform"):
        return 0

    np.random.seed(0)
    test_data = np.random.normal(1, 0.5, 100)

    # test if we correctly jump to a higher likelihood state
    kwargs = {
        "prior_func": uniform_log_prior,
        "likelihood_func": log_likelihood,
        "prior_mode": "uniform",
    }
    hopefully_true = metropolis(0.5, 0.99, test_data, **kwargs)
    assert hopefully_true is True, "failed to accept jump to higher likelihood state"

    # make sure we have the propoer ratio of jumps to a well known lower likelihood state, over 10000 samples
    test_list = np.zeros(10000)
    np.random.seed()
    for i in range(len(test_list)):
        test_list[i] = metropolis(1, 0.9, test_data, **kwargs)

    ratio = sum(test_list) / len(test_list)
    assert ratio > 0.42, "rejected too many samples"
    assert ratio < 0.48, "accepted too many samples"

    return "its an old code, but it checks out"
