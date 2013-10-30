#
#
#
#
#
#
#
#


import numpy as np
import matplotlib.pyplot as plt


def calc_d1(start, finish, air, h, water1, feed1, water2, feed2):
    '''
    calculates evolving isotope ratios (d18O per mil) in blood given
    information about air, blood, water and feed isotope ratios, and
    blood turnover time, from initial conditions.

    inputs:     start (day of experiment start, float)
                finish (day of experiment end, float)
                air (d18O of air, per mil)
                h (blood half life, days)
                water1 (d18O of initial water, per mil)
                feed1 (d18O of initial feed, per mil)
                water2 (d18O of switch water, per mil)
                feed2 (d18O of switch feed, per mil)

    outputs:    dvalue (vector of d18O over time, per mil in days)
    '''

    # Initial and equilibrium blood d18O
    bloodA = (.60*water1 + .30*feed1 + .10*air) # starting blood d18O
    bloodB = (.60*water2 + .30*feed2 + .1*air) # blood equilibrium d18O

    # create time series (days) and vector for d18O
    t = np.linspace(start, finish, num=(finish-start+1)) # time t in days
    dvalue = np.empty(finish-start+1) # empty vector for isotope ratio

    # calculate changing d18O over time
    for d in dvalue:
        dvalue = ((bloodA - bloodB)*(np.exp( (-(np.log(2))/h)*t))) + bloodB

    finish = t[-1]
    bloodA = dvalue[-1]

    return (dvalue, finish, bloodA, water2)


def calc_d2(start, finish, air, h, bloodA, water2, feed2):
    '''
    calculates evolving isotope ratios (d18O per mil) in blood given
    information about air, blood, water and feed isotope ratios, and
    blood turnover time, after initial conditions
        
    inputs:     start (day of experiment start, float)
                finish (day of experiment end, float)
                air (d18O of air, per mil)
                h (blood half life, days)
                water2 (d18O of switch water, per mil)
                feed2 (d18O of switch feed, per mil)
        
    outputs:    dvalue (vector of d18O over time, per mil in days)
    '''
    
    # Initial and equilibrium blood d18O
    bloodB = (.60*water2 + .30*feed2 + .1*air) # blood equilibrium d18O
    
    # create time series (days) and vector for d18O
    t = np.linspace(start, finish, num=(finish-start+1)) # time t in days
    dvalue = np.empty(finish-start+1) # empty vector for isotope ratio
    
    # calculate changing d18O over time
    for d in dvalue:
        dvalue = ((bloodA - bloodB)*(np.exp( (-(np.log(2))/h)*t))) + bloodB
    
    finish = t[-1]
    bloodA = dvalue[-1]
    
    return (dvalue, finish, bloodA, water2)

def main():

    # phase 1: start, finish, air, h, water1, feed1, water2, feed2
    dvalue, finish, bloodA, water2 = calc_d1(0., 9., 0., 14., 0., -3., 0., -3.)

    dvalue1 = dvalue
    finish1 = finish
    blood1 = bloodA
    waterA = np.empty(finish1+1); waterA.fill(water2)
    
    # phase 2: start, finish, air, h, bloodA, water3, feed3
    dvalue, finish, bloodA, water2 = calc_d2(0., 19., 0., 14., blood1, 11., -3.)

    dvalue2 = dvalue
    finish2 = finish
    blood2 = bloodA
    waterB = np.empty(finish2+1); waterB.fill(water2)

    # phase 3: start, finish, air, h, bloodA, water4, feed4
    dvalue, finish, bloodA, water2 = calc_d2(0., 41., 0., 14., blood2, 0., -3.)

    dvalue3 = dvalue
    finish3 = finish
    blood3 = bloodA
    waterC = np.empty(finish3+1); waterC.fill(water2)

    # phase 4: start, finish, air, h, bloodA, water5, feed5
    dvalue, finish, bloodA, water2 = calc_d2(0., 19., 0., 14., blood3, 11., -3.)

    dvalue4 = dvalue
    finish4 = finish
    blood4 = bloodA
    waterD = np.empty(finish4+1); waterD.fill(water2)

    # phase 5: start, finish, air, h, bloodA, water6, feed6
    dvalue, finish, bloodA, water2 = calc_d2(0., 79., 0., 14., blood4, 0., -3.)

    dvalue5 = dvalue
    finish5 = finish
    blood5 = bloodA
    waterE = np.empty(finish5+1); waterE.fill(water2)

    # append blood d18O from all phases together into one history
    
    ab = np.append(dvalue1, dvalue2)
    bc = np.append(ab, dvalue3)
    cd = np.append(bc, dvalue4)
    d18O_history = np.append(cd, dvalue5)
   
    # append drinking water d18O from all phases into one history
   
    ab = np.append(waterA, waterB)
    bc = np.append(ab, waterC)
    cd = np.append(bc, waterD)
    water_history = np.append(cd, waterE)

    # create history for feed, air
    feed_history = np.empty(finish1+finish2+finish3+finish4+finish5+5); feed_history.fill(-3)
    air_history = np.empty(finish1+finish2+finish3+finish4+finish5+5); air_history.fill(2)

    # plot blood d18O over time

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(d18O_history, c='r', alpha=1, label='blood')
    ax.plot(water_history, c='b', alpha=1, label='water')
    ax.plot(feed_history, c='g', alpha=1, label='feed')
    ax.plot(air_history, c='y', alpha=1, label='air')
    ax.set_title(b'Blood d18O per mil over time in days')
    ax.legend(loc='best')
    plt.show()


    return 0

if __name__ == '__main__':
    main()







