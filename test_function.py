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

    """ Testing the likelihood function on fake Supernovae data.

    This function test the likelihood function on fake Supernovae
    data that is created for a given set of the cosmological 
    parameters in the same format as the real SN data file. We
    calculate the likelihood over a range of Omega_M and Omega_lambda
    parameters and then verify that the maximum of the log likelihood
    function corresponds to the Omega_M and Omega_lambda values used
    to create the fake data.
    """
    # The cosmological parameters used to create the fake data.

    test_params = [0.3, 0.6, 75, -19.23]

    # We create a redshift array with 200 datapoints that takes
    # values starting at redshift of 0.002 and goes up to 2.

    fake_data = pd.DataFrame(np.linspace(0.002, 2.0, 200), columns=["zcmb"])

    # Adding the 'dmb' column to the fake data that mimics
    # the statistical error. We choose the error to be very low.

    fake_data["dmb"] = np.random.uniform(0, 0.01, len(fake_data.zcmb))

    mean = mag_model(test_params, fake_data.zcmb)

    cov = np.diag(pow(pd.Series.to_numpy(fake_data.dmb), 2))

    # Calculate the apparent magnitude values for each redshift in the
    # fake data using a normal distribution such that mean is calculated
    # using the apparent magnitude function and the variance is given by
    # the square of the fake statistical error given by the 'dmb' column.

    fake_data["mb"] = np.random.multivariate_normal(mean, cov)

    # Create an array of the Omega_M and Omega_lambda values over which
    # the log likelihood values are calculated.

    omega_arr = np.arange(0.1, 1.0, 0.01)

    likelihood_mat = np.zeros((len(omega_arr), len(omega_arr)))

    # Calculate the log likelihood values over all values of Omega_M and
    # Omega_lambda.

    for i, omega_m_item in enumerate(omega_arr):
        for j, omega_l_item in enumerate(omega_arr):
            likelihood_mat[i, j] = likelihood(
                [omega_m_item, omega_l_item, test_params[2], test_params[3]], fake_data
            )
        print("{:2.1%} done".format(i / len(omega_arr)), end="\r")

    # Find the value of Omega_M and Omega_lambda for which the log likelihood
    # is the highest.

    max_omega_m = omega_arr[int(np.argmax(likelihood_mat) / len(omega_arr))]
    max_omega_l = omega_arr[int(np.argmax(likelihood_mat) % len(omega_arr))]

    # The max values found above should be close to the Omega_M and Omega_lambda
    # values used to create a fake data.

    assert all(
        np.isclose(
            [max_omega_m, max_omega_l], [test_params[0], test_params[1]], atol=0.01
        )
    ), "Likelihood function does not work"

    print("Likelihood function works well on the fake data.")


def likelihood_test_contour_plot():

    """Testing the likelihood function 

    This function does a brute force likelihood sweep of the omegas
    setting M=74 and H0=-19.23. It might not be a 'unit test' 
    (in that there is no simple assert statement), but we found it 
    to be invaluable when diagnosing bugs.

    returns
    ------
    Shows the log likelihood contour plot.
    """

    # import Supernovae data

    data = pd.read_csv("lcparam_DS17f.txt", sep=" ")

    # Create an array of the Omega_M and Omega_lambda values over which
    # the log likelihood values are calculated.

    resolution = 80
    p1_min = 0.1
    p1_max = 1.2
    p2_min = 0.5
    p2_max = 1.15

    p1 = np.linspace(p1_min, p1_max, resolution)
    p2 = np.linspace(p2_min, p2_max, resolution)

    x, y = np.meshgrid(p1, p2)

    # Calculate the log likelihood values over all values of Omega_M and
    # Omega_lambda.

    z = np.zeros((resolution, resolution))

    for i, p1_item in enumerate(p1):
        for j, p2_item in enumerate(p2):
            z[i, j] = likelihood([p1_item, p2_item, 74, -19.23], data)

    # Plotting the contour plot

    c1 = np.max(z) - 2.3 / 2  # the value corresponding to 68% CI.
    c2 = np.max(z) - 6.17 / 2  # the value corresponding to 95% CI.

    levels = [c2, c1, np.max(z)]  # Defining the contour levels

    contour = plt.contourf(x, y, z, levels)

    cbar = plt.colorbar(contour)

    cbar.set_ticklabels(["95% CI", "68% CI", ""])

    plt.title("Log likelihood contour plot when H0=74 and M=-19.23 are set")

    plt.show()


def metropolis_test():

    """ Testing the metropolis function.

    Unit test for the metropolis part of the codebase. For this test, we choose a
    uniform prior and we choose the likelihood function to be a gaussian. We then 
    create a fake data set where we know the answer by setting the numpy seed=0.
    Using this data set, we first verify that metropolis returns True when the 
    proposed jump is to a higher likelihood region. Then, we verify that the jumps
    to a lower likelihood region have approximately the correct acceptance proportion
    over 10,000 trials. The correct proportion was calibrated to be ~45.5%, but it 
    allows a small region around that, because we are taking finite samples.

    """

    def log_likelihood(data, param):
        return -0.5 * np.sum((data - param) ** 2)

    def uniform_log_prior(params, magnitude_mode="uniform"):
        return 0

    # Creating a fake dataset (add the reasoning for seeding)
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

    np.random.seed()

    # Check if we have the proper ratio of jumps to a well known lower likelihood
    # state, over 10000 samples.

    # Finding the expected acceptance ratio.

    asymptotic_acceptance_prob = np.exp(
        log_likelihood(test_data, 0.7) - log_likelihood(test_data, 1)
    )

    # Finding the acceptance ratio for the fake data

    test_list = np.zeros(500000)

    for i in range(len(test_list)):
        test_list[i] = metropolis(1, 0.7, test_data, **kwargs)

    ratio = sum(test_list) / len(test_list)

    # Checking if the acceptance ratio for the fake data is close to the
    # expected ratio with a tolerance of 10%. (add the reasoning for tolerance)

    assert ratio > 0.9 * asymptotic_acceptance_prob, "rejected too many samples"
    assert ratio < 1.1 * asymptotic_acceptance_prob, "accepted too many samples"

    return "The metropolis function works well."


def chain_test():

    """ Testing the mcmc chain.

    This runs the mcmc chain, setting all paramters but the Omega_M fixed
    and checks if the value of Omega_M converges to a value that 
    corresponds to the maximum log likelihood value when all the other 
    parameters remain the same. 
    """

    # import Supernovae data

    data = pd.read_csv("lcparam_DS17f.txt", sep=" ")

    # Running the mcmc chain and outputting the values at which
    # the chain converges.

    _, _, convergence_value = chain(
        data,
        20000,
        400,
        0.01,
        start_state=[0.2, 0.82, 74, -19.23],
        gen_variances=[0.05, 0, 0, 0],
        prior_mode="uniform",
    )

    # Creating an array of Omega_M values over which the log likelihood
    # values are calculated.

    om_arr = np.linspace(0.1, 1.0, 100)

    lik_arr = np.zeros(100)

    # Calculating the log likelihood values.

    for i in range(100):
        lik_arr[i] = likelihood([om_arr[i], 0.82, 74, -19.23], data)

    # Finding the value of Omega_M corresponding to the max likelihood.

    max_omega_m = om_arr[np.argmax(lik_arr)]

    # Checking if mcmc chain converged to the correct Omega_M value.

    assert np.isclose(
        convergence_value[0], max_omega_m, atol=0.01
    ), "The chain failed to converge towards max likelihood value for Omega_M"

    print("The mcmc chain finds the correct value for Omega_M")


def mcmc_lambda_cdm_test():

    """ Testing the mcmc chain.

    This function tests the mcmc chain for the lambda CDM model
    which has only three parameters omega_M, H0 and M. We know
    from Scolnic et al 18 that the the Omega_M value for the
    lambda CDM model should converge to 0.284 +/- 0.012. 
    
    All the functions for the lambda CDM model is available in
    the lambda_cdm_function.py file. I would recommend check that
    you check out that file for a better understanding of this 
    test function.
    """

    # import Supernovae data

    data = pd.read_csv("lcparam_DS17f.txt", sep=" ")

    # Running the mcmc chain and outputting the values at which
    # the chain converges. We input the lambda cdm prior and
    # likelihood function for the prior and likelihood function
    # input to this chain.

    _, _, convergence_value = chain(
        data,
        20000,
        1500,
        0.005,
        start_state=[0.8, 75, -19.23],
        gen_variances=[0.1, 1.0, 0],
        prior_func=lambda_cdm_prior,
        likelihood_func=lambda_cdm_likelihood,
        prior_mode="uniform",
    )

    # The value of Omega_M at which the chain converged should be
    # equal to 0.284 with an error of 0.012.

    assert np.isclose(
        convergence_value[0], 0.284, atol=0.012
    ), "The chain doesn't converge to the right Omega_M value"

    print("mcmc chain works perfectly for the lambda cdm model")
