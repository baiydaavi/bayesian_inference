import numpy as np
from prior_likelihood import prior
from prior_likelihood import likelihood

def metropolis(params, candidate_params, data, prior_func, likelihood_func, prior_mode = 'uniform'):  
    if prior_func(candidate_params, magnitude_mode=prior_mode)==0:
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
    data is the data set
    max_trials is to prevent it from taking too long if it gets stuck without convergence
    convergence_window is how large a range it averages over for convergence
    convergence_threshhold is the maximum allowed percent change for reaching convergence 
    start_state is the initial values for all parameters: np array of length 4
    variances is the variance for each generating gaussian: np array of length 4
    likelihood and prior func are the functions for likelihood and prior
    prior_mode is the prior mode
    '''
    chain=[]
    rejects=[]
    current = start_state
    i=0
    convergence = False
    while convergence == False and i < max_trials:
        candidate = np.random.normal(current,variances)
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