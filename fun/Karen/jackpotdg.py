# for Karen.

import numpy as np


def load(fname):
    return np.genfromtxt(fname, dtype='u2')

def print_over_size(data, threshold):
    list_over_threshold = []
    for i,j in zip(data, xrange(data.size)):
        if i > 100:
            list_over_threshold.append((i,j))
    return list_over_threshold

def lawnmower(data, threshold):
    mowed_data = []
    for i in data:
        if i <= threshold:
            mowed_data.append(i)
        elif i > threshold:
            mowed_data.append(i/15)
    return mowed_data

def logging_data(data):
    data[data==0] = 1
    logged = np.rint(np.log(data))
    logged = logged.astype(int)
    return logged.tolist()

def save_file(data_to_save, new_file_name):
    np.savetxt(new_file_name, data_to_save, fmt='%s', delimiter=',')

def main():

    data = load('data.txt')
    threshold = 10000
    list_over_threshold = print_over_size(data, threshold)
    mowed_data = lawnmower(data, threshold)
    logged_data = logging_data(data)

    #save_file(mowed_data, 'test01.txt')

    return 0

if __name__ == '__main__':
    main()
