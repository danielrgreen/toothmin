# compare1

import numpy as np
import matplotlib.pyplot as plt

def main():

    isoshape = (34, 5)
    old_data = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 11.58, 11.39, 13.26, 12.50, 11.88, 9.63, 13.46, 12.83, 11.60, 12.15, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 10.38, 13.13, 13.37, 12.41, 13.31, 13.77, 13.51, 13.53, 13.41, 13.57, 13.99, 13.61, 13.43, 13.40, 12.40, 12.94, 12.43, 12.10, 11.13, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 11.00, 0.00, 0.00, 0.00, 0.00, 12.08, 12.91, 13.11, 12.70, 12.69, 12.23, 12.56, 11.53, 12.82, 12.36, 12.51, 10.69, 11.33, 13.33, 13.12, 13.21, 13.07, 13.76, 12.90, 14.63, 11.81, 9.76, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 12.21, 11.04, 12.81, 12.20, 12.69, 12.31, 12.44, 12.12, 10.84, 12.85, 12.90, 13.13, 13.74, 13.18, 11.91, 12.53, 13.10, 12.28, 12.92, 10.95, 12.83, 13.20, 13.25, 12.10, 11.95, 12.08, 11.65, 8.45, 0.00, 0.00, 0.00, 13.01, 12.39, 12.05, 12.25, 13.42, 12.68, 11.84, 12.43, 10.19, 11.24, 10.55, 11.33, 12.09, 12.56, 13.71, 12.03, 10.78, 12.75, 12.67, 12.50, 12.48, 12.50, 11.96, 12.21, 12.28, 9.88, 11.85, 12.44, 11.07, 11.18, 10.68, 11.42, 12.39, 10.08])
    new_data = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 11.58, 11.39, 13.26, 12.50, 11.88, 9.63, 13.46, 12.83, 11.60, 12.15, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 13.13, 13.37, 12.41, 13.31, 13.77, 13.51, 13.53, 13.41, 13.57, 13.99, 13.61, 13.43, 13.40, 12.40, 12.94, 12.43, 12.10, 11.13, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 11.00, 0.00, 0.00, 0.00, 0.00, 12.08, 12.91, 10.38, 13.29, 13.36, 12.85, 13.15, 12.35, 13.31, 12.89, 12.92, 13.35, 13.12, 13.21, 13.08, 13.30, 13.67, 12.45, 11.82, 11.32, 11.81, 9.76, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 12.21, 11.04, 12.81, 12.20, 12.69, 13.00, 13.07, 13.11, 12.98, 13.20, 13.37, 13.24, 12.26, 12.61, 13.19, 12.50, 13.01, 12.75, 13.08, 12.97, 13.15, 12.52, 12.33, 12.08, 11.87, 11.07, 11.65, 8.45, 0.00, 0.00, 0.00, 13.01, 12.39, 12.05, 12.25, 13.42, 12.68, 11.84, 12.43, 12.86, 12.69, 12.95, 12.66, 12.89, 13.52, 12.47, 12.91, 12.95, 12.87, 12.41, 12.72, 12.82, 12.38, 12.44, 12.89, 11.03, 12.63, 12.99, 13.13, 12.43, 7.35, 12.10, 11.42, 12.39, 10.08])
    diff_data = np.absolute(old_data - new_data)

    old_data = np.reshape(old_data, (isoshape[1],isoshape[0]))
    new_data = np.reshape(new_data, (isoshape[1],isoshape[0]))
    diff_data = np.reshape(diff_data, (isoshape[1],isoshape[0]))

    old_data[old_data==0.] = np.nan
    new_data[new_data==0.] = np.nan

    new_data_reduced1 = np.ma.masked_array(new_data, np.isnan(new_data))
    new_data_reduced2 = np.mean(new_data_reduced1, axis=0)
    new_data_reduced = new_data_reduced2.filled(np.nan)

    old_data_reduced1 = np.ma.masked_array(old_data, np.isnan(old_data))
    old_data_reduced2 = np.mean(old_data_reduced1, axis=0)
    old_data_reduced = old_data_reduced2.filled(np.nan)

    collapsed = np.hstack((old_data_reduced, new_data_reduced)).reshape(2,34)

    tooth = np.array([13.87, 13.23, 12.64, 0., 0., 0., 14.13, 13.86, 12.98, 12.62, 12.94, 12.88, 0., 0., 12.79, 12.67, 12.22, 0., 0., 0., 0., 12.69, 12.15, 11.86]).reshape(4,6)


    fig = plt.figure(dpi=100)
    
    ax1 = plt.subplot(4,1,1)
    ax1.set_title('original data')
    cimg1 = ax1.imshow(np.flipud(old_data), aspect='auto', interpolation='nearest', origin='lower',
                        vmin=7., vmax=14)
    cax1 = fig.colorbar(cimg1)

    ax2 = plt.subplot(4,1,2)
    ax2.set_title('new data')
    cimg2 = ax2.imshow(np.flipud(new_data), aspect='auto', interpolation='nearest', origin='lower',
                        vmin=7., vmax=14)
    cax2 = fig.colorbar(cimg2)

    ax3 = plt.subplot(4,1,3)
    ax3.set_title('difference')
    cimg3 = ax3.imshow(np.flipud(diff_data), aspect='auto', interpolation='nearest', origin='lower',
                        vmin=0., vmax=3.)
    cax3 = fig.colorbar(cimg3)

    ax4 = plt.subplot(4,1,4)
    ax4.set_title('collapsed: old above, new below')
    cimg4 = ax4.imshow(np.flipud(tooth), aspect='auto', interpolation='nearest', origin='lower',
                        vmin=7., vmax=14.)
    cax4 = fig.colorbar(cimg4)


    fig.subplots_adjust(hspace=.5)
    plt.show()



    return 0

if __name__ == '__main__':
    main()






