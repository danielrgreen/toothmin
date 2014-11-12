# Propagating 2D waves
# Daniel Green, 2014
# min_equation.py
#
'''
# X(x) = Ax np.cos(KxX) + Bx np.sin(KxX)
# Y(y) =
Ay np.cos(KyY) + By np.sin(KyY)
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

def parabolic(days, length, height,
                 As, Cs, Ps, Os, Ls, angle_s,
                 Am, Cm, Pm, Om, Lm, angle_m, day_ct):

    # exp( -(length - speed*days)**2 /2)

    half = np.percentile(days, 50) + 0.5
    print half

    v = (-((day_ct-(half+1))/half)**2 + 1)
    print 'v', v
    dx = np.cumsum(v)
    print '1', dx
    dx = np.hstack([0., dx[:-1]])
    print '2', dx
    dx = dx[days]
    print '3', dx
    
    
    s = (-((days-(half+1))/half)**2 + 1.5)
    print s[:,1,1]
    
    q = (
           As*np.exp( -( (length - angle_s*height) - dx*(days-Os))**2 / (2*Ps))
         + Am*np.exp( -( (length - angle_m*height) - dx*(days-(Om+Os)))**2 / (2*Pm))
        ) 

    return q


def wave_kill(days, length, height,
                 As, Cs, Ps, Os, Ls,
                 Am, Cm, Pm, Om, Lm):
    q = ((As*np.sin(np.pi*(Cs*(length-Os)-(days))/Ps)+As) * np.exp((Cs*length-(days))/Ls)) #+ ((Am*np.sin(np.pi*(Cm*(length-(height+Om))-(days-60))/Pm)+Am) * np.exp((Cs*length-(days-60))/Lm))

    return q

def simplest_wave(days, As, Cs, Ps):

    q = np.sin(np.pi*(Cs*days)/Ps)
    q[0:12] = -1.
    q[28:60] = -1.
    print days.shape, q.shape

    fig = plt.figure(figsize=(4,4), dpi=100)
    ax = fig.add_subplot(1,1,1)
    ax.plot(days, q, '--', linewidth=2)
    plt.show()


def exp_wave(days, length, height,
                 As, Cs, Ps, Os, Ls, angle_s,
                 Am, Cm, Pm, Om, Lm, angle_m):

    # exp( -(length - speed*days)**2 /2)
    q = (
           As*np.exp( -( (length - angle_s*height) - Cs*(days-Os))**2 / (2*Ps))
         + Am*np.exp( -( (length - angle_m*height) - Cm*(days-(Om+Os)))**2 / (2*Pm))
        ) 

    return q

def picture(grid):

    fig = plt.figure(figsize=(4,4), dpi=100)
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(grid.T, interpolation='none')
    ax.set_title(b'A mature tooth', fontsize=10)
    #ax.set_ylim(0, 70)
    #ax.set_xlim(0., 90)
    ax.set_ylabel('Tooth height', fontsize=10)
    ax.set_xlabel('Tooth length', fontsize=10)
    cax = fig.colorbar(im)

    plt.show()
    
def plot_stack_sequentially(gridstack, day_ct):
    
    # Plot array stacks with depth, colorbar
    vmax = np.max(gridstack)
    for i,j in enumerate(day_ct):
        print 'plotting image %d' % j
        fig = plt.figure(figsize=(4,4), dpi=100)
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(gridstack[i,:,:].T, vmax=vmax)
        ax.set_title(r'$ \mathrm{image} \ \mathrm{no.} \ %d$' % j, fontsize=10)
        cax = fig.colorbar(im)
        fig.savefig('para_min_small01_%d_days.png' % j)
        plt.close()

def main():

    days = 100. # number of mineralization days
    height = 10. # height of enamel crown
    length = 50. # length of enamel crown
    angle_s = 1.6
    angle_m = -.12
    day_ct = np.arange(days)

    As = 0.2 # amplitude of secretion wave
    Cs = 1. # speed of secretion wave (can also influence angle)
    Ps = 8. # period of secretion wave
    Ls = 15. # duration of secretion wave
    Os = angle_s * height # Offset

    Am = 0.15 # ampitude of maturation wave
    Cm = 1. # speed of maturation wave
    Pm = 18. # period of maturation wave
    Lm = 40. # duration of maturation wave
    Om = 10. # Offset
    
    idx_grid = np.indices((days, length, height))
    d = idx_grid[0]
    l = idx_grid[1]
    h = idx_grid[2]
    
    tooth = parabolic(d, l, h, As, Cs, Ps, Os, Ls, angle_s, Am, Cm, Pm, Om, Lm, angle_m, day_ct)
    mature = np.sum(tooth, axis=0)

    picture(mature)
    plot_stack_sequentially(tooth, day_ct)

    return 0

if __name__ == '__main__':
    main()
