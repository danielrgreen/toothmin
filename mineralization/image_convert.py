# Image convert
#
#
#
#
#
#

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.special
import argparse, sys, warnings
from os.path import abspath, expanduser
import os, os.path
import Image

def load_image(fname):
    img = plt.imread(abspath(fname))
    return img



def main():
    parser = argparse.ArgumentParser(prog='enamelsample',
                  description='Resample image of enamel to standard grid',
                  add_help=True)
    parser.add_argument('images', type=str, nargs='+', help='Images of enamel.')

    if 'python' in sys.argv[0]:
        offset = 2
    else:
        offset = 1
    args = parser.parse_args(sys.argv[offset:])

    warnings.simplefilter('ignore')

    # Load each tooth image
    
    for i,fname in enumerate(args.images):
        print 'Processing %s ...' % fname
        
        img = load_image(fname)

        
        #img[img==0.] = np.nan
        img = img * 65536.
        img = (.000046269664 * img) -0.07135082343
        img = (5382.821966 * img) + 628.547584
        img[np.isnan(img)] = 0.
        #img = img / 65536.
        img = Image.fromarray(img)
        img.mode = 'I'
        img.point(lambda i:i*(1/256.)).convert('L').save('%s new.png' % fname)
    
        
    return 0

if __name__ == '__main__':
    main()


