import numpy as np
from prior_likelihood import prior
from prior_likelihood import likelihood
import matplotlib.pyplot as plt

def metropolis(params, candidate_params, data, prior_func, likelihood_func, prior_mode = 'uniform'): 
    '''
    this is the function that decides if we keep trials or not, it is called inside of the mcmc chain algorithm

    parameters
    ----------
    params : list of dimensions expected by likelihood_func/prior_func
        parameter values we are currently at

    candidate_params : list of dimensions expected by likelihood_func/prior_func
        potential new parameters values to move to

    data :  whatever format data that likelyhood_function needs as an input
        the dataset we are running the MCMC on
    
    prior_func : function with inputs (params, magnitude_mode=arg)
        function that calculates the prior probability of our set of parameters

    likelihood_func : function with inputs: (params, data)
        function that calculates the likelyhood of our set of parameters given the data

    prior_mode : an input that is recognized by the prior function as mag_mode=priorm_mode

    returns
    -------
    True : if we should accept the move to candidate params
    False : if we shou;d reject the move to candidate params

    ''' 

    if prior_func(candidate_params, magnitude_mode=prior_mode)=='forbidden':
        return False

    else:
        def get_log_prob(params):
            if prior_mode == 'uniform':
                return likelihood_func(params, data)
            else:
                return prior_func(params, magnitude_mode=prior_mode)+likelihood_func(params, data)

        threshhold = np.exp(min(0,get_log_prob(candidate_params)-get_log_prob(params)))

        decide=np.random.uniform(0,1,1)
    
        if threshhold > decide:
            return True
        else:
            return False

def chain(data, max_trials=10000, convergence_window=50, convergence_threshhold=.001, start_state= np.ones(4)+1, variances=np.ones(4)/5,
    prior_func=prior, likelihood_func=likelihood, prior_mode='uniform'):
    '''
    this is the core function that makes our MCMC chain, it relies on the metropolis and convergence_test functions defined in this document


    parameters
    ----------

    data :  whatever format data that likelyhood_function needs as an input
        the dataset we are running the MCMC on

    max_trials : int
        prevents it from taking too long if it gets stuck without convergence

    convergence_window : int
        how large a range it averages over for convergence

    convergence_threshhold : number>0 and < 1
        the maximum allowed percent change for reaching convergence, .01 means 1%

    start_state : list of dimensions expected by likelihood/prior functions
        initial values of all

    variances : None, 1-D list or array, or 2D numpy array
        sets the variance for generating new samples using np.random.multivariate_normal
        if None: uses a hardcoded non-diagonal covariance matrix
        if 1-D list or array : uses a diagonal covariance matrix with diagonal elements = list elements
        if 2-D array : uses the 2D array as the covariance matrix 

    prior_func : function with inputs (params, magnitude_mode=arg)
        function that calculates the prior probability of our set of parameters

    likelihood_func : function with inputs: (params, data)
        function that calculates the likelyhood of our set of parameters given the data

    prior_mode : an input that is recognized by the prior function as mag_mode=priorm_mode

    returns
    -------
    chn : numpy array of dimension [N, number of parameters]
        this is your MCMC chain, N is  2*convergence window< N< mat_trials
    rej : numpy array of dimension [N, number of parameters]
        these are the samples that got rejected by the algorithm,
        will have np.nan for the whole row if the trial was accepted
    '''
    chain=[]
    rejects=[]
    current = start_state
    i=0
    convergence = False

    if variances is None:
        covariance = .1*np.array([
            [.015, .024, .070, 0.0],
            [.024, .048, .177, 0.0],
            [.070, .177, 1, 0.0],
            [0.0, 0.0, 0.0, 0.0]
            ])
    else:
        if len(np.shape(variances)) == 1:
            covariance = np.diag(variances)
        if len(np.shape(variances)) == 2:
            covariance = variances

    while convergence == False and i < max_trials:
        candidate = np.random.multivariate_normal(current, covariance)
        i += 1
        if metropolis(current, candidate, data, prior_func, likelihood_func, prior_mode= prior_mode):
            rejects.append(np.zeros(4)*np.nan)
            current = candidate
        else:
            rejects.append(candidate)
        chain.append(current)
        
        convergence, diff_booleans = convergence_test(chain, convergence_window, convergence_threshhold)
        
    rej= np.asarray(rejects)
    chn= np.asarray(chain)
    
    print('total trials:{}. accepted {:.1f}% of trials'.format(i, 100*(1-sum(rej[:,0] > 0)/i)))
    if convergence == False:
        print('convergence failed. converged parameters:', diff_booleans)
    return chn, rej

def convergence_test(chain, convergence_window, convergence_threshhold):
    '''
    this function exists solely to be called inside of the chain function, and it does a simple convergence test
    where we compare the average of the parameters over two non-overlapping windows of our most recent chain data
    and claim convergence when those avergaes are equal to each other within a tolerance: convergence_threshold

    parameters
    ----------

    chain : a list of parameter values, dimension (some int, number of params )
        the mcmc chain we are testing
    
    convergence_window : int
        how large a range it averages over for convergence

    convergence_threshhold : number>0 and < 1
        the maximum allowed percent change for reaching convergence, .01 means 1%
    
    returns
    -------

    True or False : boolean
        True if the mean of the most recent L samples is within the threshold % of the mean over the previous L samples for all params
        False if not any of the avergaes has changed by more than the threshold %

    diff_booleans : 1-D list of True/False, length equal to number of params
        True/False depending if the corresponding parameter has converged or not
    '''
    if len(chain) > 2*convergence_window:
        old_means = np.mean(chain[-2*convergence_window + 1: -convergence_window], axis=0)
        new_means = np.mean(chain[-convergence_window: -1], axis=0)
        diff_booleans = abs(new_means - old_means)/abs(old_means) < convergence_threshhold

        if sum(diff_booleans) == len(diff_booleans):
            return True, diff_booleans

        else:
            return False, diff_booleans
    else:
        return False, []

#ignore this function, it was just a weird test there will be no documentation for it.
def asynchronous_chain(data, max_trials=10000, convergence_window=50, convergence_threshhold=.001, start_state= np.ones(4)+1, variances=np.ones(4)/5,
    prior_func=prior, likelihood_func=likelihood, prior_mode='uniform'):

    chain=[]
    current = start_state
    i=0
    convergence = False

    while i < max_trials and convergence==False:

        candidate = np.random.normal(current,variances)
        i += 1

        start_index = int(np.random.uniform(0,4))
        for k in range(4):
            j = (start_index+k)%4
            temporary = np.copy(current)
            temporary[j] = candidate[j]
            if metropolis(current, temporary, data, prior_func, likelihood_func, prior_mode= prior_mode):
                current = temporary

        chain.append(current)
        
        convergence, diff_booleans = convergence_test(chain, convergence_window, convergence_threshhold)
        

    chn= np.asarray(chain)
    print('total trials:{}'.format(i))
    if convergence == False:
        print('convergence failed. converged parameters:', diff_booleans)
    return chn


def plot_chain_behaviour(chain, rejects, plot_rejects=True, one_d_hist_1=0, one_d_hist_2=1, two_d_hist_1=0, two_d_hist_2=1, one_d_bins=30, two_d_bins=50, two_d_histogram=True, save=False):
    '''
    this function is for plotting trace plots of all 4 parameters, and 1-D/2D histograms of w/e 2 paramters we want.

    parameters
    ----------
    chain : numpy array
        the chain we are plottinh
    rejects : numpy array, same shape as chain
        the rejected samples from the chain
    plot_refects = True/False
        False if you dont want to plot rejects
    one_d_his_1, one_d_his_2, two_d_his1, two_d_his_2 : ints
        these are the indices of the parameters you want to plot in the histograms (default is 0,1 for the two omegas)
    one_d_bins, two_d_bins : ints
        number of bins for our 1d and 2d histograms
    two_d_histogram : True/False
        if False, we plot a scatterplot instead of histogram
    save : True/False
        True for save, False for no save

    returns
    -------
    shows and/or saves plots, no returns
    '''
    od1 = one_d_hist_1
    od2 = one_d_hist_2
    td1 = two_d_hist_1
    td2 = two_d_hist_2
    names = dict([(0,'$\\Omega_m$'),(1,'$\\Omega_\\Lambda$'),(2,'$M$'),(3,'$H_0$')])

    fig,ax = plt.subplots(3, 2, figsize=(20,15))

    hist_or_scatter = dict ([(True, 'histogram'), (False, 'scatter plot')])

    fig.suptitle('plots 1-4 are trace plots, 5 is a 1D historgram of 1 or 2 parameters and 6 is a 2D'+hist_or_scatter[two_d_histogram])

    ax[0,0].plot(chain[:,0])
    ax[0,0].set_title(names[0])
    ax[0,1].plot(chain[:,1])
    ax[0,1].set_title(names[1])
    ax[1,0].plot(chain[:,2])
    ax[1,0].set_title(names[2])
    ax[1,1].plot(chain[:,3])
    ax[1,1].set_title(names[3])

    if plot_rejects == True:
        rej_alpha = 400/len(rejects[:,0])
        ax[0,0].plot(rejects[:,0], '+', alpha=rej_alpha)
        ax[0,1].plot(rejects[:,1], '+', alpha=rej_alpha)
        ax[1,0].plot(rejects[:,2], '+', alpha=rej_alpha)
        ax[1,1].plot(rejects[:,3], '+', alpha=rej_alpha)

    cutoff = int(len(chain[:,0])/5)
    cchn = chain[cutoff:,:]    

    mean_names = dict([(0,'$\\overline{\\Omega}_m$'),(1,'$\\overline{\\Omega}_\\Lambda$'),(2,'$\\overline{M}$'),(3,'$\\overline{H}_0$')])

    ax[2,0].hist(cchn[:,od1],bins=one_d_bins, density=1)
    ax[2,0].axvline(np.mean(cchn[:,od1]), color='k')
    ax[2,0].text(np.mean(cchn[:,0]),-3, mean_names[od1]+'={:.3f}'.format(np.mean(cchn[:,0])))

    if od2 is not None:

        ax[2,0].hist(cchn[:,1],bins=one_d_bins, density=1)
        ax[2,0].axvline(np.mean(cchn[:,od2]), color='k')
        ax[2,0].text(np.mean(cchn[:,1]),-3,mean_names[od2]+'={:.3f}'.format(np.mean(cchn[:,1])))

        p_range = np.array([[min(cchn[:,td1]), max(cchn[:,td1])], [min(cchn[:, td2]), max(cchn[:, td2])]])
        ex_range = np.zeros((2, 2))
        L = .2*(p_range[:,1]-p_range[:, 0])
        ex_range[:,0], ex_range[:,1]  = p_range[:, 0] - L, p_range[:, 1] + L

    if two_d_histogram:
        ax[2,1].hist2d(cchn[:,td1],cchn[:,td2], bins=two_d_bins, range=[[ex_range[0,0], ex_range[0,1]], [ex_range[1,0], ex_range[1,1] ]])
    else:
        ax[2,1].scatter(cchn[:,td1],cchn[:,td2], alpha=.05 )

    ax[2,1].set_xlabel(names[td1])
    ax[2,1].set_ylabel(names[td2])

    if save:
        plt.savefig('chain{}.png'.format(len(chain[:,0])))

    plt.show()


def likelihood_test(data, resolution, p1_min, p1_max, p2_min, p2_max, p1_slice=.52, p2_slice=.82, two_d=True, save=False):
    '''
    this function does a brute force likelihood sweep of the omegas setting M=74 and H0=-19.23.

    params
    -----
    data: pandas data frame
        this is the data
    resolution : int
        number of points to sample on each axis
    p1_min, p1_max, p2_min, p2_max : numbers in the domain of the respective omega
        the min/max value of the omegas we want to test
    p1_slice(p2_slice) : numbers in the domain of omega_m(omega_lambda)
        this is the value we hold constant of each parameter when making the 1d plots
    two_d : True/False
        if false, we only do 1D plots. will be much faster
    save: True/False
        True for save, False for no save
    
    returns
    ------
    shows and/or saves plots. no returns
    '''
    p1= np.linspace(p1_min,p1_max,resolution)
    p2= np.linspace(p2_min,p2_max,resolution)
    p1_lik=np.zeros(resolution)
    p2_lik=np.zeros(resolution)

    names = dict([(0,'$\\Omega_m$'),(1,'$\\Omega_\\Lambda$')])

    for i, p1_item in enumerate(p1):
            p1_lik[i]= likelihood([p1_item,p2_slice,74,-19.23], data)
    for j,p2_item in enumerate(p2):
        p2_lik[j] = likelihood([p1_slice,p2_item,74,-19.23], data)

    if two_d:    
        fig, ax = plt.subplots(1,3, figsize=(18,5))
    else:
        fig, ax = plt.subplots(1,2, figsize=(12,5))

    for item in ax:
        item.set_xlabel(names[0])
        item.set_ylabel(names[1])

    ax[0].plot(p1, p1_lik)
    ax[0].set_title('log likelihood of p1 when p2={}'.format(p2_slice))
    ax[1].plot(p2, p2_lik)
    ax[1].set_title('log likelihood of p2 when p1={}'.format(p1_slice))
    
    if two_d:
        x, y = np.meshgrid(p1,p2)
        z=np.zeros((resolution,resolution))
        for i, p1_item in enumerate(p1):
            for j,p2_item in enumerate(p2):
                z[i,j]= likelihood([p1_item,p2_item,74,-19.23], data)
    
        c1 = np.max(z)- 2.3/2
        c2 = np.max(z)- 6.17/2
        contours=[c2, c1, np.max(z)]
        c = ax[2].contour(x, y, z, contours)
        fig.colorbar(c, ax=ax[2])
        ax[2].set_title('log_likelihood when H0=74 and M=-19.23 are set')
    plt.show()
    if save:
        plt.savefig('likelihood_test{}'.format(resolution))
