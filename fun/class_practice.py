# Practice in object oriented coding
# Gregory, Daniel September 2014


import numpy as np
import matplotlib.pyplot as plt





class Person:

    def __init__(self, name, age, employment, born):
        self.name = name
        self._age = age
        self.employment = employment
        self.born = born

    def hello(self):
        print 'Hello, my name is ' + self.name

    def __call__(self):
        print 'You called?!'

    def age(self):
        print 'I am %d years old' % self._age

    def work(self):
        print 'I am a ' + self.employment

    def origin(self):
        print 'I am from ' + self.born

    def aging(self):
        self._age += 1

    def get_age(self):
        return self._age

    def __repr__(self):
        txt = 'I swear I really am from %s' % self.born
        txt += 'And I actually named &s' % self.name

class Randomwalk:

    def __init__(self, dimensions=2):
        self.n_dim = dimensions
        self.x = [np.random.random(self.n_dim)]

    def step(self):
        self.x.append(self.x[-1] + (2. * (np.random.random(self.n_dim) -.495)))

    def get_path(self):
        return np.array(self.x)




def main():

    #jim = Person('Jim', 21, 'student', 'Minneapolis')
    #jim.hello()
    #jim()
    #jim.age()
    #jim.work()
    #jim.origin()
    #jim.aging()
    #jim.age()

    walk = Randomwalk(dimensions=3)
    for i in xrange(10000):
        walk.step()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    x = walk.get_path()
    
    ax.plot(x[:,0], x[:,1], c='k', ls='-', alpha=.1)
    ax.scatter(x[:,0], x[:,1], c=x[:,2],
               linewidths=1., alpha=0.1)
    plt.show()

    return 0

if __name__ == "__main__":
    main()
