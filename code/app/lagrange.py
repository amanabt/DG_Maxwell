#! /usr/bin/env python3

import numpy as np
import arrayfire as af
from scipy import special as sp

from utils import utils
from app import params

def LGL_points(N):
    '''
    Calculates : math: `N` Legendre-Gauss-Lobatto (LGL) points.
    LGL points are the roots of the polynomial 
    :math: `(1 - \\xi ** 2) P_{n - 1}'(\\xi) = 0`
    Where :math: `P_{n}(\\xi)` are the Legendre polynomials.
    This function finds the roots of the above polynomial.

    Parameters
    ----------
    N : int
        Number of LGL nodes required
    
    Returns
    -------
    lgl : arrayfire.Array [N 1 1 1]
          An array consisting of the Lagrange-Gauss-Lobatto Nodes.
                          
    Reference
    ---------
    `https://goo.gl/KdG2Sv`
    '''
    xi                 = np.poly1d([1, 0])
    legendre_N_minus_1 = N * (xi * sp.legendre(N - 1) - sp.legendre(N))
    lgl_points         = legendre_N_minus_1.r
    lgl_points.sort()
    lgl_points         = af.np_to_af_array(lgl_points)

    return lgl_points


def lobatto_weights(n, x):
    '''
    Calculates the weight function for an index :math:`n`
    and points :math: `x`.
    
    :math::
        `w_{n} = \\frac{2 P(x)^2}{n (n - 1)}`,
        Where P(x) is $ (n - 1)^th $ index.
    
    Parameters
    ----------
    n : int
        Index for which lobatto weight function
    
    x : arrayfire.Array [N 1 1 1]
        1D array of points where weight function is to be calculated.
    
    
    Returns
    -------
    gauss_lobatto_weights : arrayfire.Array
                          An array of lobatto weight functions for
                          the given :math: `x` points and index.
    Reference
    ---------
    Gauss-Lobatto weights Wikipedia link-
    `https://goo.gl/o7WE4K`
    '''
    P = sp.legendre(n - 1)

    gauss_lobatto_weights = (2 / (n * (n - 1)) / (P(x))**2)

    return gauss_lobatto_weights


def lagrange_basis_coeffs(x):    
    '''
    A function to get the coefficients of the Lagrange basis polynomials for
    a given set of x nodes.
    
    It calculates the Lagrange basis polynomials using the formula:
    :math::
    `L_i = \\prod_{m = 0, m \\notin i}^{N - 1}\\frac{(x - x_m)}{(x_i - x_m)}`

    Parameters
    ----------
    x : numpy.array
        An array consisting of the :math: `x` nodes using which the
        lagrange basis functions need to be evaluated.

    Returns
    -------
    lagrange_basis_poly : numpy.ndarray
                          A :math: `N \\times N` matrix containing the
                          coefficients of the Lagrange basis polynomials such
                          that :math:`i^{th}` lagrange polynomial will be the
                          :math:`i^{th}` row of the matrix.
    '''
    X = np.array(x)
    lagrange_basis_poly = np.zeros([X.shape[0], X.shape[0]])
    
    for j in np.arange(X.shape[0]):
        lagrange_basis_j = np.poly1d([1])
        
        for m in np.arange(X.shape[0]):
            if m != j:
                lagrange_basis_j *= np.poly1d([1, -X[m]]) \
                                    / (X[j] - X[m])
        lagrange_basis_poly[j] = lagrange_basis_j.c
    
    return lagrange_basis_poly


def lagrange_basis(i, x):
    '''
    Calculates the value of the :math: `i^{th}` Lagrange basis (calculated
    using the :math:`\\xi_{LGL}` points) at the :math: `x` coordinates.
    
    Parameters
    ----------
    
    i : int
        The Lagrange basis which is to be calculated.
    
    x : arrayfire.Array
        The coordinates at which the `i^{th}` Lagrange polynomial is to be
        calculated.
    
    Returns
    -------
    
    l_xi_j : arrayfire.Array
             Array of values of the :math: `i^{th}` Lagrange basis
             calculated at the given :math: `x` coordinates.
    '''
    x_tile      = af.transpose(af.tile(x, 1, params.N_LGL))
    power       = utils.linspace((params.N_LGL-1), 0, params.N_LGL)
    power_tile  = af.tile(power, 1, x.shape[0])
    x_pow       = af.arith.pow(x_tile, power_tile)
    l_xi_j      = af.blas.matmul(params.lBasisArray[i], x_pow)
    
    return l_xi_j


def lagrange_basis_function(lagrange_coeff_array):
    '''
    Funtion to calculate the value of lagrange basis functions over LGL
    nodes.
    
    Returns
    -------
    L_i    : arrayfire.Array [N 1 1 1]
             The value of lagrange basis functions calculated over the LGL
             nodes.
    '''
    xi_tile    = af.transpose(af.tile(params.xi_LGL, 1, params.N_LGL))
    power      = af.flip(af.range(params.N_LGL))
    power_tile = af.tile(power, 1, params.N_LGL)
    xi_pow     = af.arith.pow(xi_tile, power_tile)
    index      = af.range(params.N_LGL)
    L_i        = af.blas.matmul(lagrange_coeff_array[index], xi_pow)
    
    return L_i


def dLp_xi_LGL(lagrange_coeff_array):
    '''
    Function to calculate :math: `\\frac{d L_p(\\xi_{LGL})}{d\\xi}`
    as a 2D array of :math: `L_i' (\\xi_{LGL})`. Where i varies along rows
    and the nodes vary along the columns.
    
    Returns
    -------
    dLp_xi        : arrayfire.Array [N N 1 1]
                    A 2D array :math: `L_i (\\xi_p)`, where i varies
                    along dimension 1 and p varies along second dimension.
    '''
    differentiation_coeffs = (af.transpose(af.flip(af.tile\
                             (af.range(params.N_LGL), 1, params.N_LGL)))
                             * lagrange_coeff_array)[:, :-1]
    
    nodes_tile         = af.transpose(af.tile(params.xi_LGL, 1, params.N_LGL - 1))
    power_tile         = af.flip(af.tile(af.range(params.N_LGL - 1), 1, params.N_LGL))
    nodes_power_tile   = af.pow(nodes_tile, power_tile)
    
    dLp_xi = af.blas.matmul(differentiation_coeffs, nodes_power_tile)
    
    return dLp_xi



def Integrate(integrand, N_quad, scheme):
    '''
    '''
    if (scheme == 'gauss_quadrature'):
        gaussian_nodes = gauss_nodes(N_quad)
        gauss_weights  = af.np_to_af_array(np.zeros([N_quad]))
        
        for i in range(0, N_quad):
            gauss_weights[i] = gaussian_weights(N_quad, i)
        value_at_gauss_nodes = af.np_to_af_array(integrand(gaussian_nodes))
        Integral = af.sum(value_at_gauss_nodes * gauss_weights)
        
    if (scheme == 'lobatto_quadrature'):
        lobatto_nodes = LGL_points(N_quad)
        lobatto_weights  = lobatto_weight_function(N_quad)
        value_at_lobatto_nodes = af.np_to_af_array(integrand(lobatto_nodes))
        Integral = af.sum(value_at_lobatto_nodes * lobatto_weights)
    
    
    return Integral
