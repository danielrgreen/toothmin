
import numpy as np


samples = 4
isotope_history = np.arange(24)


resampled = []

for i in isotope_history:
    d = np.random.normal(i, .1, samples)
    resampled = np.append(resampled, d)
    
resampled = np.reshape(resampled, (isotope_history.size, samples))


'''
sigma = np.random.normal(size=(a.size, samples))

sigma = np.einsum('ij,i->ij', sigma, a)
'''

print resampled
