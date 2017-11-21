#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
import numpy as np
import os
import h5py
from tqdm import trange
from matplotlib import pyplot as pl

from dg_maxwell import wave_equation
from dg_maxwell import msh_parser
from dg_maxwell import wave_equation_2d
from dg_maxwell import isoparam
from dg_maxwell import lagrange
from dg_maxwell import params
from dg_maxwell import utils

def A_matrix():
    '''
    '''

    A_ij = wave_equation_2d.A_matrix(params.N_LGL) / 100

    return A_ij

def volume_integral(u):
    '''
    Vectorize, p, q, moddims.
    '''
    dxi_dx   = 10.
    deta_dy  = 10.
    jacobian = 100.
    c_x = params.c_x
    c_y = params.c_y


    dLp_xi_ij_Lq_eta_ij  = params.dLp_xi_ij_Lq_eta_ij
    dLq_eta_ij_Lp_xi_ij  = params.dLq_eta_ij_Lp_xi_ij


    volume_integrand_ij_1 = c_x * dxi_dx * af.broadcast(utils.multiply,\
                                    dLp_xi_ij_Lq_eta_ij,\
                                    u) / jacobian

    volume_integrand_ij_2 = c_y * deta_dy * af.broadcast(utils.multiply,\
                                    dLq_eta_ij_Lp_xi_ij,\
                                    u) / jacobian

    volume_integrand_ij = af.moddims(volume_integrand_ij_1 + volume_integrand_ij_2, params.N_LGL ** 2,\
                                     (params.N_LGL ** 2) * 100)

    lagrange_interpolation = af.moddims(wave_equation_2d.lag_interpolation_2d(volume_integrand_ij, params.N_LGL),
                                        params.N_LGL, params.N_LGL, params.N_LGL ** 2  * 100)


    volume_integrand_total = utils.integrate_2d_multivar_poly(lagrange_interpolation[:, :, :], N_quad = 9, scheme = 'gauss')
    volume_integral        = af.transpose(af.moddims(volume_integrand_total, 100, params.N_LGL ** 2))

    return volume_integral


def lax_friedrichs_flux(u):
    '''
    '''
    u = af.reorder(af.moddims(u, params.N_LGL ** 2, 10, 10), 2, 1, 0)

    diff_u_boundary = af.np_to_af_array(np.zeros([10, 10, params.N_LGL ** 2]))

    u_xi_minus1_boundary_right   = u[:, :, :params.N_LGL]
    u_xi_minus1_boundary_left    = af.shift(u[:, :, -params.N_LGL:], d0=0, d1 = 1)
    u[:, :, :params.N_LGL] = (u_xi_minus1_boundary_right + u_xi_minus1_boundary_left) / 2

    diff_u_boundary[:, :, :params.N_LGL] = (u_xi_minus1_boundary_right - u_xi_minus1_boundary_left)

    u_xi_1_boundary_left  = u[:, :, -params.N_LGL:]
    u_xi_1_boundary_right = af.shift(u[:, :, :params.N_LGL], d0=0, d1=-1)
    u[:, :, :params.N_LGL]     = (u_xi_minus1_boundary_left + u_xi_minus1_boundary_right) / 2

    diff_u_boundary[:, :, -params.N_LGL:] = (u_xi_minus1_boundary_right - u_xi_minus1_boundary_left)


    u_eta_minus1_boundary_down = af.shift(u[:, :, params.N_LGL - 1:params.N_LGL ** 2:params.N_LGL], d0=-1)
    u_eta_minus1_boundary_up   = u[:, :, 0:-params.N_LGL + 1:params.N_LGL]
    u[:, :, 0:-params.N_LGL + 1:params.N_LGL] = (u_eta_minus1_boundary_down\
                                               + u_eta_minus1_boundary_up) / 2
    diff_u_boundary[:, :, 0:-params.N_LGL + 1:params.N_LGL] = (u_eta_minus1_boundary_up\
                                                               -u_eta_minus1_boundary_down)



    u_eta_1_boundary_down = u[:, :, params.N_LGL - 1:params.N_LGL ** 2:params.N_LGL]
    u_eta_1_boundary_up   = af.shift(u[:, :, 0:-params.N_LGL + 1:params.N_LGL], d0=1)

    u[:, :, params.N_LGL - 1:params.N_LGL ** 2:params.N_LGL] = (u_eta_1_boundary_up\
                                                              +u_eta_1_boundary_down) / 2

    diff_u_boundary[:, :, params.N_LGL - 1:params.N_LGL ** 2:params.N_LGL] = (u_eta_1_boundary_up\
                                                                             -u_eta_1_boundary_down)



    u = af.moddims(af.reorder(u, 2, 1, 0), params.N_LGL ** 2, 100)
    diff_u_boundary = af.moddims(af.reorder(diff_u_boundary, 2, 1, 0), params.N_LGL ** 2, 100)
    F_xi_e_ij  = params.c_x * u - params.c_lax_2d_x * diff_u_boundary
    F_eta_e_ij = params.c_y * u - params.c_lax_2d_x * diff_u_boundary

    return F_xi_e_ij, F_eta_e_ij


def surface_term_vectorized(u):
    '''
    '''
    N_LGL = params.N_LGL
    lagrange_coeffs = params.lagrange_coeffs

    xi_LGL  = params.xi_LGL
    eta_LGL = params.xi_LGL

    # Values for dx/dxi and dy/deta for the particular case.
    dx_dxi = 0.1
    dy_deta = 0.1

    f_xi_surface_term  = lax_friedrichs_flux(u)[0]
    f_eta_surface_term = lax_friedrichs_flux(u)[1]

    Lp_xi   = af.moddims(af.reorder(af.tile(utils.polyval_1d(params.lagrange_coeffs,
                            xi_LGL), 1, 1, params.N_LGL), 1, 2, 0), params.N_LGL, 1, params.N_LGL ** 2)

    Lq_eta  = af.tile(af.reorder(utils.polyval_1d(params.lagrange_coeffs,\
                         eta_LGL), 1, 2, 0), 1, 1, params.N_LGL)

    Lp_xi_1      = af.moddims(af.reorder(af.tile(utils.polyval_1d(lagrange_coeffs, xi_LGL[-1]),\
                           1, 1, params.N_LGL), 2, 1, 0), 1, 1, params.N_LGL ** 2)
    Lp_xi_minus1 = af.moddims(af.reorder(af.tile(utils.polyval_1d(lagrange_coeffs, xi_LGL[0]),\
                           1, 1, params.N_LGL), 2, 1, 0), 1, 1, params.N_LGL ** 2)

    Lq_eta_1      = af.moddims(af.tile(af.reorder(utils.polyval_1d(lagrange_coeffs,\
                            eta_LGL[-1]), 0, 2, 1), 1, 1, params.N_LGL), 1, 1, params.N_LGL ** 2)
    Lq_eta_minus1 = af.moddims(af.tile(af.reorder(utils.polyval_1d(lagrange_coeffs,\
                             eta_LGL[0]), 0, 2, 1), 1, 1, params.N_LGL), 1, 1, params.N_LGL ** 2)


    # xi = 1 boundary
    Lq_eta_1_boundary   = af.broadcast(utils.multiply, Lq_eta, Lp_xi_1)
    Lq_eta_F_1_boundary = af.broadcast(utils.multiply, Lq_eta_1_boundary, f_xi_surface_term[-params.N_LGL:, :])
    Lq_eta_F_1_boundary = af.reorder(Lq_eta_F_1_boundary, 0, 3, 2, 1)


    lag_interpolation_1 = af.sum(af.broadcast(utils.multiply, lagrange_coeffs, Lq_eta_F_1_boundary), 0)
    lag_interpolation_1 = af.reorder(lag_interpolation_1, 2, 1, 3, 0)
    lag_interpolation_1 = af.transpose(af.moddims(af.transpose(lag_interpolation_1),\
                                       params.N_LGL, params.N_LGL ** 2 * 100))

    surface_term_pq_xi_1 = lagrange.integrate(lag_interpolation_1) * dy_deta
    surface_term_pq_xi_1 = af.moddims(surface_term_pq_xi_1, params.N_LGL ** 2, 100)

    # xi = -1 boundary
    Lq_eta_minus1_boundary   = af.broadcast(utils.multiply, Lq_eta, Lp_xi_minus1)
    Lq_eta_F_minus1_boundary = af.broadcast(utils.multiply, Lq_eta_minus1_boundary, f_xi_surface_term[:params.N_LGL, :])
    Lq_eta_F_minus1_boundary = af.reorder(Lq_eta_F_minus1_boundary, 0, 3, 2, 1)

    lag_interpolation_2 = af.sum(af.broadcast(utils.multiply, lagrange_coeffs, Lq_eta_F_minus1_boundary), 0)
    lag_interpolation_2 = af.reorder(lag_interpolation_2, 2, 1, 3, 0)
    lag_interpolation_2 = af.transpose(af.moddims(af.transpose(lag_interpolation_2),\
                                       params.N_LGL, params.N_LGL ** 2 * 100))

    surface_term_pq_xi_minus1 = lagrange.integrate(lag_interpolation_2) * dy_deta
    surface_term_pq_xi_minus1 = af.moddims(surface_term_pq_xi_minus1, params.N_LGL ** 2, 100)


    # eta = -1 boundary
    Lp_xi_minus1_boundary   = af.broadcast(utils.multiply, Lp_xi, Lq_eta_minus1)
    Lp_xi_F_minus1_boundary = af.broadcast(utils.multiply, Lp_xi_minus1_boundary,\
                                           f_eta_surface_term[0:-params.N_LGL + 1:params.N_LGL])
    Lp_xi_F_minus1_boundary = af.reorder(Lp_xi_F_minus1_boundary, 0, 3, 2, 1)

    lag_interpolation_3 = af.sum(af.broadcast(utils.multiply, lagrange_coeffs, Lp_xi_F_minus1_boundary), 0)
    lag_interpolation_3 = af.reorder(lag_interpolation_3, 2, 1, 3, 0)
    lag_interpolation_3 = af.transpose(af.moddims(af.transpose(lag_interpolation_3),\
                                       params.N_LGL, params.N_LGL ** 2 * 100))

    surface_term_pq_eta_minus1 = lagrange.integrate(lag_interpolation_3) * dx_dxi
    surface_term_pq_eta_minus1 = af.moddims(surface_term_pq_eta_minus1, params.N_LGL ** 2, 100)


    # eta = 1 boundary
    Lp_xi_1_boundary   = af.broadcast(utils.multiply, Lp_xi, Lq_eta_1)
    Lp_xi_F_1_boundary = af.broadcast(utils.multiply, Lp_xi_1_boundary,\
                                           f_eta_surface_term[params.N_LGL - 1:params.N_LGL ** 2:params.N_LGL])
    Lp_xi_F_1_boundary = af.reorder(Lp_xi_F_1_boundary, 0, 3, 2, 1)

    lag_interpolation_4 = af.sum(af.broadcast(utils.multiply, lagrange_coeffs, Lp_xi_F_1_boundary), 0)
    lag_interpolation_4 = af.reorder(lag_interpolation_4, 2, 1, 3, 0)
    lag_interpolation_4 = af.transpose(af.moddims(af.transpose(lag_interpolation_4),\
                                       params.N_LGL, params.N_LGL ** 2 * 100))

    surface_term_pq_eta_1 = lagrange.integrate(lag_interpolation_4) * dx_dxi
    surface_term_pq_eta_1 = af.moddims(surface_term_pq_eta_1, params.N_LGL ** 2, 100)

    surface_term_e_pq = surface_term_pq_xi_1\
                      - surface_term_pq_xi_minus1\
                      + surface_term_pq_eta_1\
                      - surface_term_pq_eta_minus1

    return surface_term_e_pq


def b_vector(u):
    '''
    '''
    b = volume_integral(u) - surface_term_vectorized(u)

    return b

def RK4_timestepping(A_inverse, u, delta_t):
    '''
    Implementing the Runge-Kutta (RK4) method to evolve the wave.

    Parameters
    ----------
    A_inverse : arrayfire.Array[N_LGL N_LGL 1 1]
                The inverse of the A matrix which was calculated
                using A_matrix() function.

    u         : arrayfire.Array[N_LGL N_Elements 1 1]
                u at the mapped LGL points

    delta_t   : float64
                The time-step by which u is to be evolved.

    Returns
    -------
    delta_u : arrayfire.Array [N_LGL N_Elements 1 1]
              The change in u at the mapped LGL points.
    '''

    k1 = af.matmul(A_inverse, b_vector(u))
    k2 = af.matmul(A_inverse, b_vector(u + k1 * delta_t / 2))
    k3 = af.matmul(A_inverse, b_vector(u + k2 * delta_t / 2))
    k4 = af.matmul(A_inverse, b_vector(u + k3 * delta_t    ))

    delta_u = delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return delta_u

def time_evolution():
    '''
    '''
    # Creating a folder to store hdf5 files. If it doesn't exist.
    results_directory = 'results/2d_hdf5_%02d' %(int(params.N_LGL))
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    A_inverse = af.np_to_af_array(np.linalg.inv(np.array(A_matrix())))
    u         = params.u_e_ij
    delta_t   = params.delta_t_2d

    for i in trange(2001):
        #    u += RK4_timestepping(A_inverse, u, delta_t)
    
        #Implementing second order time-stepping.
        u_n_plus_half =  u + af.matmul(A_inverse, b_vector(u))\
                              * delta_t / 2
    
        u            +=  af.matmul(A_inverse, b_vector(u_n_plus_half))\
                          * delta_t
    
        L1_norm = af.mean(af.abs(params.u_e_ij - u))
        if (L1_norm >= 100):
            break
        if (i % 10) == 0:
            h5file = h5py.File('results/2d_hdf5_%02d/dump_timestep_%06d' %(int(params.N_LGL), int(i)) + '.hdf5', 'w')
            dset   = h5file.create_dataset('u_i', data = u, dtype = 'd')
    
            dset[:, :] = u[:, :]
        
        if (i > 1999):
            print(L1_norm)
        if (i == 1000):
            print(L1_norm)
        if (i == 1001):
            print(L1_norm)
    return L1_norm


#def contour_2d(u, index):
#    '''
#    '''
#    color_levels = np.linspace(-0.1, 1.1, 100)
#    u_plot = af.flip(af.moddims(u, params.N_LGL, params.N_LGL, 10, 10), 0)
#    x_plot = af.flip(af.moddims(params.x_e_ij, params.N_LGL, params.N_LGL, 10, 10), 0)
#    y_plot = af.flip(af.moddims(params.y_e_ij, params.N_LGL, params.N_LGL, 10, 10), 0)
#    
#    
#    x_contour = af.np_to_af_array(np.zeros([params.N_LGL * 10, params.N_LGL * 10]))
#    y_contour = af.np_to_af_array(np.zeros([params.N_LGL * 10, params.N_LGL * 10]))
#    u_contour = af.np_to_af_array(np.zeros([params.N_LGL * 10, params.N_LGL * 10]))
#    fig = pl.figure()
#    #
#    for i in range(100):
#        p = int(i / 10)
#        q = i - p * 10
#        x_contour[p * params.N_LGL:params.N_LGL * (p + 1),\
#                  q * params.N_LGL:params.N_LGL * (q + 1)] = x_plot[:, :, q, p]
#
#        y_contour[p * params.N_LGL:params.N_LGL * (p + 1),\
#                  q * params.N_LGL:params.N_LGL * (q + 1)] = y_plot[:, :, q, p]
#
#        u_contour[p * params.N_LGL:params.N_LGL * (p + 1),\
#                  q * params.N_LGL:params.N_LGL * (q + 1)] = u_plot[:, :, q, p]
#    
#    x_contour = np.array(x_contour)
#    y_contour = np.array(y_contour)
#    u_contour = np.array(u_contour)
#    pl.contourf(x_contour, y_contour, u_contour, 200, levels = color_levels, cmap = 'jet')
#    pl.gca().set_aspect('equal')
#    pl.colorbar()
#    pl.title('Time = %f' % (index * 10 * 1e-3))
#    fig.savefig('results/2D_Wave_images/%04d' %(index / 10) + '.png')
#    return