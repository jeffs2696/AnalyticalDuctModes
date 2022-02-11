#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import special as sp
from scipy.optimize import fsolve, minimize
import pychebfun
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.optimize import brentq
import sample.helpers as fcn 


def main():
    a = 0.045537                        # inner radius
    b = 0.277630                         # outer radius
    m = -10
    M = 0.28993

    roots, re_roots, im_roots, F, f_cheb = fcn.kradial(m,a,b)

    print("radial wave numbers are: ", re_roots, '\n')

    kaxial = fcn.k_axial(M,re_roots)
    print("axial wavenumbers are: ", kaxial)

    k_array = np.linspace(5,100,5000)
    plt.plot(k_array, F(k_array, m, a, b), label = '$F$')
    plt.plot(f_cheb.roots(), F(f_cheb.roots(), m, a, b), 'o', label = 'roots')
    plt.ylim(ymin=-1000.0, ymax = 1000)
    plt.legend()
    plt.grid()
    plt.xlabel('$k$'); 
    plt.show()





main()









# Method 2 for setting up the problem

# def F(kr,m,a,b):
#     Jp = 0.5*(sp.jv(m-1,kr) - sp.jv(m+1,kr))
#     Yp = 0.5*(sp.yv(m-1,kr) - sp.yv(m+1,kr))
#     fval = Jp(m,kr*a)*Yp(m,kr*b)-Jp(m,kr*b)*Yp(m,kr*a)

#     return fval.real, fval.imag




# ***The below sequence works***

# Jp = lambda m,x : 0.5*(sp.jv(m-1,x) - sp.jv(m+1,x))
# Yp = lambda m,x : 0.5*(sp.yv(m-1,x) - sp.yv(m+1,x))


# F = lambda k,m,a,b : Jp(m,k*a)*Yp(m,k*b)-Jp(m,k*b)*Yp(m,k*a)
# f_cheb = pychebfun.Chebfun.from_function(lambda x: F(x, m, a, b), domain = (5,50))

# roots = f_cheb.roots()
# print(roots)


# k_array = np.linspace(5,50,5000)
# plt.plot(k_array, F(k_array, m, a, b), label = '$F$')
# plt.plot(f_cheb.roots(), F(f_cheb.roots(), m, a, b), 'o', label = 'roots')
# plt.ylim(ymin=-.1, ymax = 0.1)
# plt.legend()
# plt.grid()
# plt.xlabel('$k$');
# **************************************************************************


#**********************************************************************************************
#****this works****
# guess=[1,5,10,15,20,25,30,35,40,45,50]
# root = opt.root(lambda k : Jp(m,k*a)*Yp(m,k*b)-Jp(m,k*b)*Yp(m,k*a), guess, method='hybr')
# print(root.x)
#************************************************************************************************
