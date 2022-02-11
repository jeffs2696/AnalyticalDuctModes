# -*- coding: utf-8 -*-

from .context import sample

import unittest


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_kradial(self):
        a = 0.045537                        # inner radius
        b = 0.277630                         # outer radius
        m = -10
        roots, re_roots, im_roots, F, f_cheb = sample.helpers.kradial(m,a,b)

        assert True
    def test_kaxial(self):
        M = 0.28993 
        a = 0.045537                        # inner radius
        b = 0.277630                         # outer radius
        m = -10
        roots, re_roots, im_roots, F, f_cheb = sample.helpers.kradial(m,a,b)
        kaxial = sample.helpers.k_axial(M,re_roots)
        assert True
    

        
 
if __name__ == '__main__':
    unittest.main()
