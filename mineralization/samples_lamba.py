# Samples to find lambda




import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# User inputs

talf = 8. #measured 
experiment_days = 20.
sample_freq = 1.
error = 0.5 # measured
start_d18O = 0.
eq_d18O = 6.
iterations = 10000
bins = 100


def calc_blood_values(fractions):

    real_blood_d18O = []
    for f in fractions:
        real_blood_d18O.append((f * start_d18O) + ((1 - f) * eq_d18O))
    real_blood_d18O = np.asarray(real_blood_d18O)
    return real_blood_d18O

def calc_blood_values_simpler(days, start_d18O, eq_d18O, talf):

    return ((start_d18O - eq_d18O)*(np.exp( (-(np.log(2))/talf)*days))) + eq_d18O

def calc_blood_values_simplest(days, talf):

    return ((start_d18O - eq_d18O)*(np.exp( (-(np.log(2))/talf)*days))) + eq_d18O

def sample_w_error(real_blood_d18O, error):

    samples = []
    for v in real_blood_d18O:
        samples.append(v + np.random.normal(0, error))
    samples = np.asarray(samples)
    return samples

def estimated_lambda(days, samples):

    popt, pcov = curve_fit(calc_blood_values_simplest, days, samples)
    calc_lambda = popt[-1]

    return calc_lambda

#def calc_likelihood(real_blood_d18O, samples):


#def calc_lambda(samples, experiment_days, sample_freq):    

        

def main():

    days = np.arange(experiment_days)
    days = days[::sample_freq]
    fractions = []
    resample = np.arange(iterations)

    for d in days:
        fractions.append(.5**(d / talf))

    real_blood_d18O = calc_blood_values(fractions)
    samples = sample_w_error(real_blood_d18O, error)
    new_samples = calc_blood_values_simplest(days, talf)

    resampled = []

    for i in resample:
        samples = sample_w_error(real_blood_d18O, error)
        resampled.append(estimated_lambda(days, samples))

    resampled = np.asarray(resampled)
    least, mean, most = np.percentile(resampled, [4, 50, 94])

    print least, mean, most

    hist, bins = np.histogram(resampled, bins=40)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2

    popt, pcov = curve_fit(calc_blood_values_simplest, days, samples)
    calc_lambdax = popt[-1]

    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.bar(center, hist, align='center', width=width)
    ax1.annotate('%i iterations \n mean = %.1f \n 95%% = %.1f - %.1f' % (iterations, mean, least, most), xy=(9,500))
    ax1.set_title('Estimated blood-water half lives')
        
    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(days, samples, 'ko', label="sampled")
    ax2.plot(days, calc_blood_values_simplest(days, *popt), 'r-', label="fitted, L = %.2f" % calc_lambdax)
    ax2.plot(days, real_blood_d18O, 'g-', label="theoretical, L = %.2f" % talf)
    ax2.legend(loc='best')
    ax2.set_title('Example of a single iteration: fitted vs. theoretical curves')
    plt.show()

    #plt.plot(days, samples, 'ko', label="sampled")
    #plt.plot(days, calc_blood_values_simplest(days, *popt), 'r-', label="fitted, L = %.2f" % calc_lambda)
    #plt.plot(days, real_blood_d18O, 'g-', label="real, L = %.2f" % talf)
    #plt.legend(loc='best')




    plt.show()


    
    return 0


if __name__ == '__main__':
    main()
