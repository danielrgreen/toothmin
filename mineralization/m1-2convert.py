__author__ = 'darouet'

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spec

def convert(m2days):
    '''
    '''
    #m2height = 75.123*spec.erf(.0028302*(m2days+70.17))+(42-75.12266) # max at 42
    m2height = 44.182*spec.erf(.003736412*(m2days+53.0767))+(40.5-44.182) # max at 40.5 optimized with full data set on nlopt
    m2height = 46.625*spec.erf(.0032506*(m2days+53.0767))+(42.46-46.625) # max at 40.5 optimized with synchrotron data set on nlopt
    m2percent = m2height / 42
    m1height = m2percent * 36
    m1days = (25000000*spec.erfinv((50*m1height-283)/1517)-1577367)/152550

    m1days_a = 163.873*spec.erfinv(0.0292948*(75.123*spec.erf(0.0028302*(m2days+70.17))-33.123)-0.0186437)-10.5459
    m1days_b = 163.873*spec.erf(0.0282485*(75.123*spec.erf(0.0028302*(m2days+70.17))-33.123)-0.0186437)-10.5459

    return m1days

def main():

    m2days = np.array([70., 80., 202., 263., 450., 500.])
    m1days = convert(m2days)

    for i in xrange(m2days.size):
        print m2days[i], m1days[i]
    print 'switch length = ', m1days[3]-m1days[2]



    return 0
if __name__ == '__main__':
    main()