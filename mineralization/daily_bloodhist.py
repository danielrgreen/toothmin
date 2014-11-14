# daily_bloodhist.py



import numpy as np
import matplotlib.pyplot as plt

def water_hist(days=365., Am=10., P=182., offset=40., mean=-7.):
    
    days = np.arange(days)
    waterhist = Am * np.sin((days-offset) * (2*np.pi) / P) + mean

    print '0', waterhist.shape
    
    return waterhist, days

def blood_hist(waterhist, feed=-8., air=-18., half_life=8.,
               feed_frac=0.3, air_frac=0.1):
    
    water_frac = 1. - feed_frac - air_frac
    air_feed = (feed_frac * feed) + (air_frac * air)
    
    remaining = .5**(1/half_life)
    bloodhist = np.empty(len(waterhist), dtype='f8')
    print 'a', bloodhist[0], 'b', water_frac, 'c', waterhist[1], 'd', air_feed
    bloodhist[0] = water_frac*waterhist[0] + air_feed
    
    for k in xrange(1, len(waterhist)):
        bloodhist[k] = remaining*bloodhist[k-1] + (1-remaining)*(water_frac*waterhist[k]+air_feed)

    return bloodhist

def picture(waterhist, bloodhist, days):

    fig = plt.figure(figsize=(4,4), dpi=100)
    ax = fig.add_subplot(1,1,1)
    ax.plot(days, waterhist, '--', c='b', linewidth=2)
    ax.plot(days, bloodhist, '-', c='r', linewidth=2)    
    plt.show()

def main():

    waterhist, days = water_hist()
    bloodhist = blood_hist(waterhist)
    picture(waterhist, bloodhist, days)


    return 0

if __name__ == '__main__':
    main()
