import numpy as np
import gvar as gv

p0 = {'p_0':1.5}
for n in range(1,20):
    p0['p_%d' %n] = 0.1

def make_p0(n,seed=None):
    np.random.seed(seed)
    p0 = {'p_0':1.5}
    for i in range(1,n+1):
        p0['p_%d' %i] = np.random.normal(0,0.5)
    return p0

priors = {'p_0':gv.gvar(1.5, 1)}
for n in range(1,20):
    priors['p_%d' %n] = gv.gvar(0,1)

def polynomial(x,p):
    ''' x = independent parameters
        p = dictionary of coefficients, p_0, p_1, ...
        y = p_0 + p_1 * x + p_2 * x**2 + ...
    '''
    y = 0
    for k in p:
        n = int(k.split('_')[1])
        y += p[k] * x**n
    return y

def add_noise(x, mu=0., sig=1., Nsamp=100, seed=None):
    ''' add random noise for all points x
        mu    : mean of noise
        sig   : width of noise
        Nsamp : number of random samples
        seed  : seed the random number generator with an int
    '''
    np.random.seed(seed)
    noise = np.random.normal(loc=mu, scale=sig, size=(Nsamp, x.shape[0]))

    return noise
