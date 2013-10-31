#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  fit_positive_function.py
#  
#  Copyright 2013 Greg Green <greg@greg-UX31A>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib as mplib

import types


class TMonotonicPointModel:
	'''
	Class encapsulating the model. It can be called as follows:
	
	    x0 = np.array([...])
	    y0 = np.array([...])
	    cov = np.array([[...],...])
	    
	    model = TModel(x0, y0, cov)
	    
	    a = np.array([...])
	    p_a = model(a)
	
	where p_a is the likelihood of the model parameters taking on values a.
	'''
	
	def __init__(self, y, sigma,
                           mu_prior=None,
                           sigma_prior=None):
		self.y = np.array(y)
		self.sigma = np.array(sigma)
		
		# Check that dimensions are correct
		if len(self.y.shape) != 1:
			raise ValueError('y must be a flat array.')
		if self.sigma.shape != self.y.shape:
			raise ValueError('''sigma must have the same shape as y.''')
		
		# Compute useful quantities
		self.n_dim = self.y.size
		self.variance = self.sigma * self.sigma
		self.inv_variance = 1. / self.variance

                if (mu_prior != None) and (sigma_prior != None):
                        self.use_prior = True
                        self.mu_prior = np.array(mu_prior)
                        self.sigma_prior = np.array(sigma_prior)
                else:
                        self.use_prior = False
	
	def __call__(self, log_Delta_y_mod, *args, **kwargs):
		y_mod = np.cumsum(np.exp(log_Delta_y_mod))
		
		Delta_y = y_mod - self.y
		
		log_likelihood = -0.5 * np.sum(Delta_y * Delta_y * self.inv_variance)
		log_prior = np.sum(log_Delta_y_mod)
                
		if self.use_prior == True:
                        Delta_y = (log_Delta_y_mod - self.mu_prior) / self.sigma_prior
                        log_prior -= 0.5 * np.sum(Delta_y * Delta_y)
                
		return log_likelihood + log_prior

	def guess(self, n_guesses, min_delta=0.001):
                # Generate a central guess
                y_0 = np.empty(self.n_dim, dtype='f8')
                
                for i in xrange(self.n_dim):
                        y_0[i] = np.max(self.y[:i+1])

                delta_y_0 = np.hstack([[y_0[0]], np.diff(y_0)])

                idx = (delta_y_0 < min_delta)
                delta_y_0[idx] = min_delta

                log_delta_y_0 = np.log(delta_y_0)

                # Add in scatter
                log_delta_y_0.shape = (1, self.n_dim)
                log_delta_y_guess = np.repeat(log_delta_y_0, n_guesses, axis=0)
                log_delta_y_guess += np.random.normal(scale=0.05, size=log_delta_y_guess.shape)

                return log_delta_y_guess



class TMCMC:
	def __init__(self, model, guess, cov, *args, **kwargs):
		self.model = model
		self.guess = np.array(guess)
		self.cov = np.array(cov)
		self.args = args
		self.kwargs = kwargs
		
		self.x = self.draw(self.guess)
		self.lnp_x = self.model(self.x, self.args, self.kwargs)
		self.N = 1
		self.sum_N = 0
		self.chain = []
		self.weight = []
		self.sum_weight = []
	
	def draw(self, x0):
		return np.random.multivariate_normal(x0, self.cov)
	
	def accept(self, y, lnp_y):
		self.flush()
		self.x = y
		self.lnp_x = lnp_y
	
	def flush(self):
		self.chain.append(self.x)
		self.weight.append(self.N)
		self.sum_N += self.N
		self.sum_weight.append(self.sum_N)
		self.N = 1
	
	def step(self):
		y = self.draw(self.x)
		lnp_y = self.model(y, self.args, self.kwargs)
		alpha = lnp_y - self.lnp_x
		if alpha >= 0.:
			self.accept(y, lnp_y)
		elif np.random.random() < np.exp(alpha):
			self.accept(y, lnp_y)
		else:
			self.N += 1
	
	def get_chain(self, burnin=0):
		if burnin == 0:
			chain = np.array(self.chain)
			weight = np.array(self.weight)
			return np.repeat(chain, weight, axis=0).T
		elif burnin >= self.sum_weight:
			raise ValueError('Burn-in is longer than length of chain.')
		
		sum_weight = np.array(self.sum_weight)
		idx = (sum_weight - burnin < 0)
		sum_weight[idx] = np.max(sum_weight)
		start = np.argmin(sum_weight)
		
		chain = np.array(self.chain[start:])
		weight = np.array(self.weight[start:])
		
		subtract = burnin
		if start > 0:
			subtract = burnin - self.sum_weight[start-1]
		weight[0] -= subtract
		
		return np.repeat(chain, weight, axis=0).T
	
	def get_stats(self, burnin=0):
		chain = self.get_chain(burnin=burnin)
		mean = np.mean(chain, axis=0)
		cov = np.cov(chain)
		return mean, cov
	
	def kernel_density_estimate(self, burnin=0):
		chain = self.get_chain(burnin=burnin)
		return stats.gaussian_kde(chain)


def test_emcee():
        import emcee

        n_points = 10
        n_walkers = 24
        n_steps = 5000
        
        # True parameters
	x = np.arange(n_points)
	Delta_y_true =  np.sin(x * 2. * np.pi / (n_points - 1.)) #np.random.random(size=n_points)
	Delta_y_true = Delta_y_true * Delta_y_true * Delta_y_true * Delta_y_true
	y_true = np.cumsum(Delta_y_true)
	
	# Generate data
	sigma = 0.1 * np.ones(n_points, dtype='f8')
	errs = sigma * np.random.normal(size=n_points)
	y_obs = y_true + errs
	
	# Initialize model
	model = TMonotonicPointModel(y_obs, sigma)
	
	# Initial guess
	Delta_y_guess = np.random.random(size=n_points)
	cov_guess = np.diag(sigma)
	
	# Set up emcee sampler
	sampler = emcee.EnsembleSampler(n_walkers, n_points, model)
        
	# Generate guesses
	guess = model.guess(n_walkers)
        
	# Sample
	pos, prob, state = sampler.run_mcmc(guess, n_steps)
	sampler.reset()
        pos, prob, state = sampler.run_mcmc(pos, n_steps)

        # Plot histograms
        '''
        for i in range(n_points):
                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                ax.hist(np.exp(sampler.flatchain[:,i]), 100, color='k', histtype='step')
                ax.set_title('Dimension %d' % i)
        '''
        
        # Plot samples
        fig = plt.figure()
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2)
        
        for sample in np.exp(sampler.flatchain[::120]):
                ax1.plot(x, sample, color='b', alpha=0.01)
        
        for sample in np.cumsum(np.exp(sampler.flatchain[::120]), axis=1):
                ax2.plot(x, sample, color='b', alpha=0.01)

        ax1.plot(x, Delta_y_true, 'k-', lw=2, alpha=0.5)
        ax2.plot(x, np.cumsum(Delta_y_true), 'k-', lw=2, alpha=0.5)
        
        plt.show()



def test_MCMC():
	n_points = 10
	
	# True parameters
	x = np.arange(n_points)
	Delta_y_true =  np.sin(x * 2. * np.pi / (n_points - 1.)) #np.random.random(size=n_points)
	Delta_y_true = Delta_y_true * Delta_y_true * Delta_y_true * Delta_y_true
	y_true = np.cumsum(Delta_y_true)
	
	# Generate data
	sigma = 0.1 * np.ones(n_points, dtype='f8')
	errs = sigma * np.random.normal(size=n_points)
	y_obs = y_true + errs
	
	# Initialize model
	model = TMonotonicPointModel(y_obs, sigma)
	
	# Initial guess
	Delta_y_guess = np.random.random(size=n_points)
	cov_guess = np.diag(sigma)
	
	# Sample using MCMC
	sampler = TMCMC(model, Delta_y_guess, cov_guess/10.)
	N = 100000
	for i in range(N):
		sampler.step()
	sampler.flush()
	
	# Get chain
	chain = np.cumsum(np.exp(sampler.get_chain(burnin=N/3)), axis=0)
	
	# Report statistics
	#mean, cov = sampler.get_stats(burnin=N/3)
	mean = np.mean(chain, axis=1)
	stddev = np.std(chain, axis=1)
	print 'Mean:'
	print mean
	print ''
	print 'Std. Deviations:'
	print stddev
	
	# Plot results
	
	'''
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	#ax.imshow(img, origin='lower', extent=(a0[0],a0[-1],a1[0],a1[-1]),
	#          aspect='auto', interpolation='bicubic', cmap='gray', alpha=0.5)
	ax.plot(chain[0], chain[1], 'g', alpha=0.1)
	#ax.scatter(chain[0], chain[1], s=sampler.weight)
	#ax.set_xlim(a0[0], a0[-1])
	#ax.set_ylim(a1[0], a1[-1])
	plt.show()
	'''
	
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	
	for c in chain.T[::100]:
		ax.plot(x, c + 1., 'k-', alpha=0.01)
		ax.plot(x, np.hstack([c[0], np.diff(c)]), 'b-', alpha=0.01)
	
	ax.errorbar(x, y_obs+1., yerr=sigma, fmt='g.')
	ax.scatter(x, Delta_y_true, c='g', s=10)
	
	plt.show()
	

def main():
        test_emcee()
        
	return 0

if __name__ == '__main__':
	main()

