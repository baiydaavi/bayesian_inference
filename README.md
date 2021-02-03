
# Bayesian Inference of Cosmological Parameters and Hubble's Constant

In this project, we will use bayesian inference to estimate the matter density parameter, the dark energy density parameter and the Hubble constant using Supernova luminosity data.

## Table of contents
* [Introduction](#introduction)
* [Usage](#usage)
* [Reading](#reading)
* [Authors](#authors)

## Introduction

This project is done as part of the Data Science for Science course (PHY 250) at UC Davis in Spring 2020. The goal of this project is to use bayesian inference to estimate cosmological parameters using Supernovae luminosity data. More specifically, we estimate the matter density parameter, the dark energy density parameter and the Hubble constant using the $\Lambda$ CDM + $\Omega_K$ model of the universe. We estimate the parameters by inferring posterior probability densities of cosmological parameters from distance vs. redshift data with the use of Markov Chain Monte Carlo. The data used is described in [Scolnic et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018ApJ...859..101S/abstract) and can be found [here](https://github.com/dscolnic/Pantheon/tree/master/Binned_data). 

## Usage

The presentation.ipynb is our 'assignment'. If you go through and execute the cells in order you will see the requested results
as well as all of our test functions.

core_mcmc_functions.py contain all the core mcmc chain functions.

prior_likelihood.py contains the prior and likelihood functions.

theoretical_mag.py contains the apparent magnitude calculator function.

test_function.py contains all the test functions.

lambda_cdm_functions.py contains all the functions needed to run the mcmc chain test on the lambda CDM model.

## Reading

Refer to the resources provided below to better your understanding of the cosmological concepts and the bayesian inference model used in this project.

### Cosmology:

1. https://phys.libretexts.org/Courses/University_of_California_Davis/UCD%3A_Physics_156_-_A_Cosmology_Workbook/A_Cosmology_Workbook/16%3A_Distance_and_Magnitude

2. https://phys.libretexts.org/Courses/University_of_California_Davis/UCD%3A_Physics_156_-_A_Cosmology_Workbook/A_Cosmology_Workbook/17%3A_Parallax%2C_Cepheid_Variables%2C_Supernovae%2C_and_Distance_Measurement

### Monte Carlo Markov Chain:

1. https://machinelearningmastery.com/markov-chain-monte-carlo-for-probability/

2. https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiVk9ml3MruAhWNvp4KHUP6CkMQFjABegQIAhAC&url=https%3A%2F%2Ftowardsdatascience.com%2Ffrom-scratch-bayesian-inference-markov-chain-monte-carlo-and-metropolis-hastings-in-python-ef21a29e25a&usg=AOvVaw3DPIB2woz5amzOtKgCyhx4

## Authors

Avinash (aavinash@ucdavis.edu), Kyle (kjray@ucdavis.edu), Pratik (pjgandhi@ucdavis.edu), Sudheer (ssreedhar@ucdavis.edu)