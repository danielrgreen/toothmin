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

start = 0.

initial_water_d18O = 0.
initial_feed_d18O = 3.
air_d18O = -4.
blood_halflife = 14. 

day_of_first_switch = 40.
length_of_first_switch = 30.
first_H2O_switch_d18O = -3.
first_feed_switch_d18O = 3.

length_of_second_switch = 50.
second_H20_switch_d18O = 0.
second_feed_switch_d18O = 3.

length_of_third_switch = 50.
third_H2O_switch_d18O = 1.
third_feed_switch_d18O = 3.

length_of_fourth_switch = 50.
fourth_H2O_switch_d18O = 0.
fourth_feed_switch_d18O = 3.

length_of_fifth_switch = 50.
fifth_H2O_switch_d18O = 0.
fifth_feed_switch_d18O = 3.

def calc_d1(start, day_of_first_switch, air_d18O, blood_half_life, initial_water_d18O, initial_feed_d18O, first_switch_d18O, second_feed_d18O):
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
    bloodA = (.60*initial_water_d18O + .30*initial_feed_d18O + .1*air_d18O) # starting blood d18O
    bloodB = (.60*first_H2O_switch_d18O + .30*first_feed_switch_d18O + .1*air_d18O) # blood equilibrium d18O

    # create time series (days) and vector for d18O
    t = np.linspace(start, day_of_first_switch, num=(day_of_first_switch - start + 1)) # time t in days
    dvalue = np.empty(day_of_first_switch - start + 1) # empty vector for isotope ratio

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

def calc_blood_hist():

    # phase 1: start, finish, air, h, water1, feed1, water2, feed2
    dvalue, finish, bloodA, water2 = calc_d1(0., 49., -2., 14., 0., -3., 0., -3.)

    dvalue1 = dvalue
    finish1 = finish
    blood1 = bloodA
    waterA = np.empty(finish1+1); waterA.fill(water2)
    
    # phase 2: start, finish, air, h, bloodA, water3, feed3
    dvalue, finish, bloodA, water2 = calc_d2(0., 29., -2., 14., blood1, 3., -1)

    dvalue2 = dvalue
    finish2 = finish
    blood2 = bloodA
    waterB = np.empty(finish2+1); waterB.fill(water2)

    # phase 3: start, finish, air, h, bloodA, water4, feed4
    dvalue, finish, bloodA, water2 = calc_d2(0., 119., -2., 14., blood2, 0., -3.)

    dvalue3 = dvalue
    finish3 = finish
    blood3 = bloodA
    waterC = np.empty(finish3+1); waterC.fill(water2)

    # phase 4: start, finish, air, h, bloodA, water5, feed5
    dvalue, finish, bloodA, water2 = calc_d2(0., 49., -2., 14., blood3, 2., -2.)

    dvalue4 = dvalue
    finish4 = finish
    blood4 = bloodA
    waterD = np.empty(finish4+1); waterD.fill(water2)

    # phase 5: start, finish, air, h, bloodA, water6, feed6
    dvalue, finish, bloodA, water2 = calc_d2(0., 109., -2., 14., blood4, 0., -3.)

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
    feed_history = np.empty(finish1+finish2+finish3+finish4+finish5+5); feed_history.fill(3.5)
    air_history = np.empty(finish1+finish2+finish3+finish4+finish5+5); air_history.fill(-3)

    return (d18O_history, water_history, feed_history, air_history)

def main():

    d18O_history, water_history, feed_history, air_history = calc_blood_hist()

    # plot blood d18O over time

    fig = plt.figure(figsize=(9,8), edgecolor='none')
    ax = fig.add_subplot(1,1,1)
    ax.plot(d18O_history, c='r', alpha=1, label='blood')
    ax.plot(water_history, c='b', alpha=1, label='water')
    ax.plot(feed_history, c='g', alpha=1, label='feed')
    ax.plot(air_history, c='y', alpha=1, label='air')
    ax.set_title(b'2b: MCMC varies water $\delta^{18}$O history to match measured tooth $\delta^{18}$O profiles')
    ax.set_ylim(-4., 4.)
    ax.set_xlim(1., 360.)
    ax.legend(loc='best')
    ax.set_ylabel(r'$ \mathrm{density} \ \mathrm{increase} \ g/cm^{3}/day$', color='r')
    ax.set_xlabel(r'$ \mathrm{cumulative} \ g/cm^{3}$', color='r')
    fig.savefig('figtitle3.png', dpi=500)
    plt.show()


    return 0

if __name__ == '__main__':
    main()







