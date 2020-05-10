import numpy as np
from prior_likelihood import prior
from prior_likelihood import likelihood
import matplotlib.pyplot as plt
import pandas as pd

from theoretical_mag import calculate_apparent_mag as mag_model


def test_mag_func_omegaK_is_0():

    test_param = [0.286, 0.714, 69.6, -19.23]

    dist_modulus = mag_model(test_param, [3])[0]

    lum_distance = round(10 ** ((dist_modulus - test_param[3]) / 5) * 10 / 10 ** 10, 2)

    expected_output = 2.59

    assert (
        lum_distance == expected_output
    ), "apparent magnitude function does not work when OmegaK = 0.0"

    print("output of the magnitude function is correct")

def test_mag_func_omegaK_is_pos():

    test_param = [0.1, 0.714, 69.6, -19.23]

    dist_modulus = mag_model(test_param, [3])[0]

    lum_distance = round(10 ** ((dist_modulus - test_param[3]) / 5) * 10 / 10 ** 10, 2)

    expected_output = 3.3

    assert (
        lum_distance == expected_output
    ), "apparent magnitude function does not work when OmegaK > 0.0"

    print("output of the magnitude function is correct")


def test_mag_func_omegaK_is_neg():

    test_param = [0.5, 0.714, 69.6, -19.23]

    dist_modulus = mag_model(test_param, [3])[0]

    lum_distance = round(10 ** ((dist_modulus - test_param[3]) / 5) * 10 / 10 ** 10, 2)

    expected_output = 2.17

    assert (
        lum_distance == expected_output
    ), "apparent magnitude function does not work when OmegaK < 0.0"

    print("output of the magnitude function is correct")

