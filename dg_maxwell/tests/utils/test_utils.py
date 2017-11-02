#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import arrayfire as af

from dg_maxwell import params
from dg_maxwell import utils
from dg_maxwell import lagrange

af.set_backend(params.backend)

def test_matmul_3D():
    '''
    '''
    M = 3
    N = 2
    P = 4
    Q = 2

    a = af.range(M * N * Q, dtype = af.Dtype.u32)
    b = af.range(N * P * Q, dtype = af.Dtype.u32)

    a = af.moddims(a, d0 = M, d1 = N, d2 = Q)
    b = af.moddims(b, d0 = N, d1 = P, d2 = Q)

    a_init = a
    b_init = b
    
    ref_a_0 = np.matmul(np.array(a_init[:, :, 0]),
                        np.array(b_init[:, :, 0]))

    ref_a_1 = np.matmul(np.array(a_init[:, :, 1]),
                        np.array(b_init[:, :, 1]))
    
    test_matmul = np.array(utils.matmul_3D(a, b))
    
    diff_mat_0 = np.abs(test_matmul[:, :, 0] - ref_a_0)
    diff_mat_1 = np.abs(test_matmul[:, :, 1] - ref_a_1)
    
    assert np.all(diff_mat_0 == 0) and np.all(diff_mat_1 == 0)



def test_poly1d_prod():
    '''
    Checks the product of the polynomials of different degrees using the
    poly1d_product function and compares it to the analytically calculated
    product coefficients.
    '''
    
    N      = 3

    N_a    = 3
    poly_a = af.range(N * N_a, dtype = af.Dtype.u32)
    poly_a = af.moddims(poly_a, d0 = N, d1 = N_a)

    N_b    = 2
    poly_b = af.range(N * N_b, dtype = af.Dtype.u32)
    poly_b = af.moddims(poly_b, d0 = N, d1 = N_b)

    ref_poly = af.np_to_af_array(np.array([[0., 0., 9., 18.],
                                           [1., 8., 23., 28.],
                                           [4., 20., 41., 40.]]))

    test_poly1d_prod = utils.poly1d_product(poly_a, poly_b)
    test_poly1d_prod_commutative = utils.poly1d_product(poly_b, poly_a)

    diff     = af.abs(test_poly1d_prod - ref_poly)
    diff_commutative = af.abs(test_poly1d_prod_commutative - ref_poly)
    
    assert af.all_true(diff == 0.) and af.all_true(diff_commutative == 0.)



def test_integrate_1d():
    '''
    Tests the ``integrate_1d`` by comparing the integral agains the
    analytically calculated integral. The polynomials to be integrated
    are all the Lagrange polynomials obtained for the LGL points.
    
    The analytical integral is calculated in this `sage worksheet`_
    
    .. _sage worksheet: https://goo.gl/1uYyNJ
    '''
    
    threshold = 1e-12
    
    N_LGL     = 8
    xi_LGL    = lagrange.LGL_points(N_LGL)
    eta_LGL   = lagrange.LGL_points(N_LGL)
    _, Li_xi  = lagrange.lagrange_polynomials(xi_LGL)
    _, Lj_eta = lagrange.lagrange_polynomials(eta_LGL)

    Li_xi  = af.np_to_af_array(Li_xi)
    Lp_xi  = Li_xi.copy()
    
    Li_Lp = utils.poly1d_product(Li_xi, Lp_xi)
    
    test_integral_gauss = utils.integrate_1d(Li_Lp, order = 9,
                                             scheme = 'gauss')
    
    test_integral_lobatto = utils.integrate_1d(Li_Lp, order = N_LGL + 1,
                                               scheme = 'lobatto')

    
    ref_integral = af.np_to_af_array(np.array([0.0333333333333,
                                               0.196657278667,
                                               0.318381179651,
                                               0.384961541681,
                                               0.384961541681,
                                               0.318381179651,
                                               0.196657278667,
                                               0.0333333333333]))
    
    diff_gauss   = af.abs(ref_integral - test_integral_gauss)
    diff_lobatto = af.abs(ref_integral - test_integral_lobatto)
    
    assert af.all_true(diff_gauss < threshold) and af.all_true(diff_lobatto < threshold)


def test_integrate_2d():
    '''
    Tests the ``integrate_2d`` by comparing the integral agains the
    analytically calculated integral. The integral to be calculated is
    
    .. math:: \\iint L_i(\\xi) L_i(\\eta) L_i(\\xi) L_i(\\eta) d\\xi d\\eta
    
    where :math:`L_i` are the Lagrange polynomials with
    :math:`i \in \{0 ... 7\}`. The analytical integral is calculated
    in this `sagews`_
    
    .. _sagews: https://goo.gl/KziEMs
    '''
    threshold = 1e-12
    
    N_LGL = 8
    xi_LGL  = lagrange.LGL_points(N_LGL)
    eta_LGL = lagrange.LGL_points(N_LGL)
    _, Li_xi  = lagrange.lagrange_polynomials(xi_LGL)
    _, Lj_eta = lagrange.lagrange_polynomials(eta_LGL)

    Li_xi  = af.np_to_af_array(Li_xi)
    Lj_eta = af.np_to_af_array(Lj_eta)
    Lp_xi  = Li_xi.copy()
    Lq_eta = Lj_eta.copy()
    
    Li_Lp = utils.poly1d_product(Li_xi, Lp_xi)
    Lj_Lq = utils.poly1d_product(Lj_eta, Lq_eta)
    
    test_gauss_integral_Li_Lp_Lj_Lq = utils.integrate_2d(Li_Lp, Lj_Lq,
                                                   order = 9,
                                                   scheme = 'gauss')

    test_lobatto_integral_Li_Lp_Lj_Lq = utils.integrate_2d(Li_Lp, Lj_Lq,
                                                           order = N_LGL + 1,
                                                           scheme = 'lobatto')

    ref_integral = af.np_to_af_array(np.array([0.00111111111111037,
                                               0.0386740852528278,
                                               0.101366575556200,
                                               0.148195388573733,
                                               0.148195388573733,
                                               0.101366575556200,
                                               0.0386740852528278,
                                               0.00111111111111037]))

    diff_gauss   = af.abs(test_gauss_integral_Li_Lp_Lj_Lq - ref_integral)
    diff_lobatto = af.abs(test_lobatto_integral_Li_Lp_Lj_Lq - ref_integral)
    
    assert af.all_true(diff_gauss < threshold) and af.all_true(diff_lobatto < threshold)
