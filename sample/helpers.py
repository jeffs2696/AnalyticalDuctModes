import pychebfun
import numpy as np
from scipy import special as sp

def get_answer():
    """Get an answer."""
    return True

def kradial(m,a,b):
    """ Compute the bessel functions as well as the zero crossings

    Inputs
    ------
    m : int
        radial mode number
    a : float     
        starting point
    b : float     
        ending point



    Outputs
    -------
    roots :
    F     :
    f_cheb:
    """
    Jp = lambda m,x : 0.5*(sp.jv(m-1,x) - sp.jv(m+1,x))
    Yp = lambda m,x : 0.5*(sp.yv(m-1,x) - sp.yv(m+1,x))
    F = lambda k,m,a,b :Jp(m,k*a)*Yp(m,k*b)-Jp(m,k*b)*Yp(m,k*a) 
    f_cheb = pychebfun.Chebfun.from_function(lambda x: F(x, m, a, b), domain = (5,100))
    re_roots = f_cheb.roots().real
    im_roots = f_cheb.roots().imag
    roots = re_roots + im_roots*1j 
    return roots, re_roots, im_roots, F, f_cheb


def k_axial(M, krad):
    freq = 5726.6
    omega = 2*np.pi*freq                      # angular frequency
    c0 = 343.15                          # speed of sound
    # rho0 = 1.225                        # density
    k_wave = omega/c0               # wave number
    beta = 1-M**2
    kaxial = (-M*k_wave + np.sqrt(k_wave**2 - beta*krad**2)) / beta**2 
    return kaxial

