# Propagating 2D waves
# Daniel Green, 2014
# min_equation.py
#
'''
# X(x) = Ax np.cos(KxX) + Bx np.sin(KxX)
# Y(y) =Ay np.cos(KyY) + By np.sin(KyY)
''' 

import numpy as np
import matplotlib.pyplot as plt

def wave(days, length, height,
         As, Cs, Ps, Os,
         Am, Cm, Pm, Om):

    q = As*np.cos(2*np.pi*Ps)

def second_wave(days, length, height,
                 As, Cs, Ps, Os,
                 Am, Cm, Pm, Om):

    q = (As*np.sin(np.pi*(Cs*(length-Os)-days)/Ps)+As) + (Am*np.sin(np.pi*(Cm*(length-(height+Om))-days)/Pm)+Am)
        
    idx = (q < 0.)
    q[idx] = 0.

    return q

def decreasing_wave(days, length,
         As, Cs, Ps, Ls,
         Am, Cm, Pm, Lm):

    q = np.sin(np.pi*(Cs*length-days)/Ps) * As*np.exp((Cs*length-days)/Ls) #+ np.cos(np.pi*(Cm*length-days)/Pm) * Am*np.exp((Cm*length-days)/Lm)
    idx = (q < 0.)
    q[idx] = 0.

    return q

def picture(grid):

    fig = plt.figure(figsize=(4,4), dpi=100)
    ax = fig.add_subplot(1,1,1)
    ax.imshow(grid.T, interpolation='none')
    ax.set_title(b'A mature tooth', fontsize=10)
    #ax.set_ylim(0, 70)
    #ax.set_xlim(0., 90)
    ax.set_ylabel('Tooth height', fontsize=10)
    ax.set_xlabel('Tooth length', fontsize=10)
    plt.show()
    
def plot_stack_sequentially(gridstack, days):
    
    # Plot array stacks with depth, colorbar
    vmax = np.max(gridstack)
    for i,j in enumerate(days):
        print j.shape
        fig = plt.figure(figsize=(4,4), dpi=100)
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(gridstack[i,:,:].T, vmax=vmax)
        ax.set_title(r'$ \mathrm{image} \ \mathrm{no.} \ %d$' % j, fontsize=10)
        cax = fig.colorbar(im)
        fig.savefig('sin_min_small7_%d_days.png' % j)

def main():

    days = 60. # number of mineralization days
    height = 10. # height of enamel crown
    length = 50. # length of enamel crown

    As = 0.2 # amplitude of secretion wave
    Cs = 1. # speed of secretion wave (can also influence angle)
    Ps = 8. # period of secretion wave
    #Ls = 15. # duration of secretion wave
    Os = np.pi / 2 # Offset
    angle = 4 # Angle of the wave

    Am = 0.15 # ampitude of maturation wave
    Cm = 1. # speed of maturation wave
    Pm = 16. # period of maturation wave
    #Lm = 40. # duration of maturation wave
    Om = np.pi / 2 # Offset
    
    idx_grid = np.indices((days, length, height))
    d = idx_grid[0]
    l = idx_grid[1]
    h = idx_grid[2]
    
    tooth = second_wave(d, l, h, As, Cs, Ps, Os, Am, Cm, Pm, Om)
    mature = np.sum(tooth, axis=0)

    print tooth.shape
    print mature.shape
    
    picture(mature)
    days = np.arange(days)
    plot_stack_sequentially(tooth, days)

    return 0

if __name__ == '__main__':
    main()
