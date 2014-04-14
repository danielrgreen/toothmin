# Making small MCMC
# Daniel Green, 2014

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mplib
import types
import emcee


class increase_model:

    def __init__(self, y_observations, sigma, mu_prior=None, sigma_prior=None):
        self.observations = np.array(y_observations)
        self.sigma = np.array(sigma)
        self.dimensions = self.observations.size
        self.variance = self.sigma**2
        self.inverse_variance = 1. / self.variance

        if (mu_prior != None) and (sigma_prior != None):
            self.use_prior = True
            self.mu_prior = np.array(mu_prior)
            self.sigma_prior = np.array(sigma_prior)
        else:
            self.use_prior = False


    def __call__(self, log_probability):
        y_model = np.cumsum(np.exp(log_probability))
        delta_y = y_model - self.observations
        log_likelihood = -0.5 * np.sum((delta_y**2)*self.inverse_variance)
        log_prior = np.sum(log_probability)
        if self.use_prior == True:
            delta_y = (log_probability - self.mu_prior) / self.sigma_prior
            log_prior -= 0.5 * np.sum(delta_y**2)

        return log_likelihood + log_prior

    def guess(self, n_guesses, min_delta=0.001):
        first_guess = np.empty(self.dimensions, dtype='f8')
        for i in xrange(self.dimensions):
                first_guess[i] = np.max(self.observations[:i+1])
        first_guess_delta = np.hstack([[first_guess[0]], np.diff(first_guess)])
        idx = (first_guess_delta < min_delta)
        first_guess_delta[idx] = min_delta
        log_first_guess_delta = np.log(first_guess_delta)
        log_first_guess_delta.shape = (1, self.dimensions)
        log_delta_y_guess = np.repeat(log_first_guess_delta, n_guesses, axis=0)
        log_delta_y_guess += np.random.normal(scale=0.01, size=log_delta_y_guess.shape)

        return log_delta_y_guess
                            

def MCMC():
    n_points = 10
    n_walkers = n_points * 4
    n_steps = 500

    x = np.arange(n_points)
    y_increase = np.sin(x * 2. * np.pi / (n_points - 1.))
    y_increase = y_increase**4
    y_sum = np.cumsum(y_increase)

    sigma = .1 * np.ones(n_points, dtype='f8')
    errors = sigma * np.random.normal(size = n_points)
    y_observations = y_increase + errors

    model = increase_model(y_observations, sigma)
    delta_y_guess = np.random.random(size=n_points)
    guess_covariance = np.diag(sigma)
    sampler = emcee.EnsembleSampler(n_walkers, n_points, model)

    guess = model.guess(n_walkers)
    pos, prob, state = sampler.run_mcmc(guess, n_steps)
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos, n_steps)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
        
    for sample in np.exp(sampler.flatchain[::120]):
        ax1.plot(x, sample, color='b', alpha=0.01)
        
    for sample in np.cumsum(np.exp(sampler.flatchain[::120]), axis=1):
        ax2.plot(x, sample, color='b', alpha=0.01)

    ax1.plot(x, y_increase, 'k-', lw=2, alpha=0.5)
    ax2.plot(x, np.cumsum(y_increase), 'k-', lw=2, alpha=0.5)
        
    plt.show()


    
def main():
        MCMC()
        
        
if __name__ == '__main__':
	main()
