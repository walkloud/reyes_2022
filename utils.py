import numpy as np

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
