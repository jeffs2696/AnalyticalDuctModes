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


# k_wave = omega/c0               # wave number
# beta = 1-M**2

# h = 0.01                        # resolution
# kr_start = 0
# kr_stop = 50
# num_steps = (kr_stop-kr_start)/h

# k_array = np.linspace(kr_start,kr_stop,num=1000)
# kr=k_array

# # The following section computes the nondimensional axial wavenumber given the radial as input
# kradial = 11.770876667533 -0.000000000j
# kaxial = (-M*k_wave + np.sqrt(k_wave**2 - beta*kradial**2)) / beta
# #print(kaxial)

# n_count = 10                    # number of radial modes
# m_count = 10                    # number of circumfrencial modes

# kmn = [m_count+1, n_count+1]


# The function I want to end up solving
# J’m(kr*a)*Y’m(kr*b) - J’m(kr*b)*Y’m(kr*a) = 0

# The following is just a check to make sure I am defining the function correctly
# Jp = lambda m,x : 0.5*(sp.jv(m-1,x) - sp.jv(m+1,x))
# Yp = lambda m,x : 0.5*(sp.yv(m-1,x) - sp.yv(m+1,x))
# F = lambda k,m,a,b : Jp(m,k*a)*Yp(m,k*b)-Jp(m,k*b)*Yp(m,k*a)

#prints the right function
# plt.plot(k_array, F(k_array,m,a,b))
# plt.ylim(ymin=-2.5, ymax = 2.5)
# plt.xlim(xmin=10.0, xmax = 40)
# plt.grid()




def main():
    a = 0.045537                        # inner radius
    b = 0.277630                         # outer radius
    m = -10
    # kr = np.linspace(5,50,1000)
    roots, re_roots, im_roots, F, f_cheb = fcn.kradial(m,a,b)
    print("radial wave numbers are: ", re_roots, '\n')
    k_array = np.linspace(5,100,5000)
    plt.plot(k_array, F(k_array, m, a, b), label = '$F$')
    plt.plot(f_cheb.roots(), F(f_cheb.roots(), m, a, b), 'o', label = 'roots')
    plt.ylim(ymin=-1000.0, ymax = 1000)
    plt.legend()
    plt.grid()
    plt.xlabel('$k$'); 
    M = 0.28993
    kaxial = fcn.k_axial(M,re_roots)
    print("axial wavenumbers are: ", kaxial)
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
