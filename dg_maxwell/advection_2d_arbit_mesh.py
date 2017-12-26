#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
import numpy as np
import os
import h5py
from tqdm import trange
from matplotlib import pyplot as pl
from tqdm import trange

from dg_maxwell import wave_equation
from dg_maxwell import msh_parser
from dg_maxwell import wave_equation_2d
from dg_maxwell import isoparam
from dg_maxwell import lagrange
from dg_maxwell import params
from dg_maxwell import utils

def A_matrix(advec_var):
    '''
    '''
    jacobian = 100.
    A_ij = wave_equation_2d.A_matrix(params.N_LGL, advec_var) / jacobian

    return A_ij



def volume_integral(u, advec_var):
    '''
    Calculates the volume integral term for the :math:`x-y` formulation of
    advection equation. This function will work for arbitrary meshes, but
    only for meshes composed of :math:`2^{nd}` order square elements with
    edge length of :math:`0.2` units in the :math:`x-y` plane. This function
    can also calculate the volume term for multiple advection equations.

    Parameters
    ----------
    u : af.Array [N_LGL^2 N_elements M 1]
        ``u_e_ij`` for :math:`M` separate equations.

    advec_var : :py:meth:`dg_maxwell.global_variables.advection_variables`
    
    Returns
    -------
    volume_integral : af.Array [N_LGL^2 M 1 1]
                      Volume integral for :math:`M` advection equations.
    '''
    shape_u_n = utils.shape(u)
    dLp_xi_ij_Lq_eta_ij = advec_var.dLp_Lq
    dLq_eta_ij_Lp_xi_ij = advec_var.dLq_Lp
    dxi_dx   = 10.
    deta_dy  = 10.
    jacobian = 100.
    c_x = params.c_x
    c_y = params.c_y

    if (params.volume_integrand_scheme_2d == 'Lobatto' and params.N_LGL == params.N_quad):
        #print('option1')
        w_i = af.flat(af.transpose(af.tile(advec_var.lobatto_weights_quadrature,
                                           1, params.N_LGL)))
        w_j = af.tile(advec_var.lobatto_weights_quadrature,
                      params.N_LGL)
        wi_wj = w_i * w_j

        wi_wj_dLp_xi = af.tile(af.broadcast(utils.multiply, wi_wj,
                                            advec_var.dLp_Lq),
                               d0 = 1, d1 = 1, d2 = 1, d3 = shape_u_n[2])

        volume_integrand_ij_1_sp = c_x * dxi_dx \
                                 * af.broadcast(utils.multiply,
                                                wi_wj_dLp_xi,
                                                af.reorder(u, d0 = 0, d1 = 1,
                                                           d2 = 3, d3 = 2)) \
                                                / jacobian

        wi_wj_dLq_eta = af.tile(af.broadcast(utils.multiply,
                                             w_i * w_j,
                                             advec_var.dLq_Lp),
                                d0 = 1, d1 = 1, d2 = 1, d3 = shape_u_n[2])

        volume_integrand_ij_2_sp = c_y * deta_dy \
                                 * af.broadcast(utils.multiply,
                                                wi_wj_dLq_eta,
                                                af.reorder(u, d0 = 0, d1 = 1,
                                                           d2 = 3, d3 = 2)) \
                                                / jacobian

        volume_integral = af.reorder(af.sum(
            volume_integrand_ij_1_sp + volume_integrand_ij_2_sp, dim = 0), \
                d0 = 2, d1 = 1, d2 = 3, d3 = 0)

    else:
        #print('option2')
        volume_integrand_ij_1 = c_x * dxi_dx * af.broadcast(utils.multiply,
                                                            af.tile(dLp_xi_ij_Lq_eta_ij,
                                                                    d0 = 1, d1 = 1, d2 = 1,
                                                                    d3 = shape_u_n[2]),
                                                            af.reorder(u, d0 = 0, d1 = 1,
                                                                       d2 = 3, d3 = 2)) \
                                                            / jacobian

        volume_integrand_ij_2 = c_y * deta_dy * af.broadcast(utils.multiply,
                                                             af.tile(dLq_eta_ij_Lp_xi_ij,
                                                                     d0 = 1, d1 = 1, d2 = 1,
                                                                     d3 = shape_u_n[2]),
                                                             af.reorder(u, d0 = 0, d1 = 1,
                                                                        d2 = 3, d3 = 2)) \
                                                             / jacobian

        volume_integrand_ij = af.moddims(volume_integrand_ij_1 + volume_integrand_ij_2,
                                         d0 = params.N_LGL ** 2,
                                         d1 = (params.N_LGL ** 2) * 100,
                                         d2 = 1,
                                         d3 = shape_u_n[2])

        volume_integrand_ij = af.moddims(af.reorder(volume_integrand_ij, d0 = 0,
                                                    d1 = 1, d2 = 3, d3 = 2),
                                         d0 = params.N_LGL * params.N_LGL,
                                         d1 = params.N_LGL * params.N_LGL \
                                            * advec_var.elements.shape[0] \
                                            * shape_u_n[2],
                                         d2 = 1)

        lagrange_interpolation = wave_equation_2d.lag_interpolation_2d(volume_integrand_ij,
                                                                       advec_var.Li_Lj_coeffs)

        volume_integrand_total = utils.integrate_2d_multivar_poly(lagrange_interpolation,
                                                                  params.N_quad,'gauss',
                                                                  advec_var)

        volume_integral = af.moddims(volume_integrand_total,
                                     d0 = params.N_LGL \
                                        * params.N_LGL \
                                        * advec_var.elements.shape[0],
                                     d1 = shape_u_n[2])

        volume_integral = af.transpose(af.moddims(volume_integral,
                                                  d0 = advec_var.elements.shape[0],
                                                  d1 = params.N_LGL ** 2,
                                                  d2 = shape_u_n[2]))

    return volume_integral



def lax_friedrichs_flux(u_n, F_n, u_n_plus_1, F_n_plus_1):
    '''
    Calculates the Lax-Friedrichs flux.
    [TODO] Documentation needed
    '''
    
    lf_flux = ((F_n_plus_1 + F_n) / 2.) \
            - (params.c_lax_2d * (u_n_plus_1 - u_n) / 2.)
    
    return lf_flux



def u_at_edge(u_e_ij, edge_id, advec_var):
    '''
    This function will work for multiple :math:`u`.
    '''
    # Function to get u at the edge of an element

    shape_u = utils.shape(u_e_ij)

    if edge_id == 0:

        u_e_ij = af.reorder(u_e_ij, d0 = 0, d1 = 3, d2 = 1, d3 = 2)
        u_e_ij = af.moddims(u_e_ij, d0 = params.N_LGL, d1 = params.N_LGL,
                            d2 = advec_var.elements.shape[0], d3 = shape_u[2])
        u_edge = af.reorder(u_e_ij[:, 0, :], d0 = 0, d1 = 2, d2 = 3, d3 = 1)

        return u_edge

    # Get bottom edge of all the elements

    if edge_id == 1:
        u_e_ij = af.reorder(u_e_ij, d0 = 0, d1 = 3, d2 = 1, d3 = 2)
        u_e_ij = af.moddims(u_e_ij, d0 = params.N_LGL, d1 = params.N_LGL,
                            d2 = advec_var.elements.shape[0], d3 = shape_u[2])
        u_edge = af.reorder(u_e_ij[0, :, :], d0 = 1, d1 = 2, d2 = 3, d3 = 0)

        return u_edge

    # Get right edge of all the elements

    if edge_id == 2:
        u_e_ij = af.reorder(u_e_ij, d0 = 0, d1 = 3, d2 = 1, d3 = 2)
        u_e_ij = af.moddims(u_e_ij, d0 = params.N_LGL, d1 = params.N_LGL,
                            d2 = advec_var.elements.shape[0], d3 = shape_u[2])
        u_edge = af.reorder(u_e_ij[:, -1, :], d0 = 0, d1 = 2, d2 = 3, d3 = 1)

        return u_edge

    # Get top edge of all the elements

    if edge_id == 3:
        u_e_ij = af.reorder(u_e_ij, d0 = 0, d1 = 3, d2 = 1, d3 = 2)
        u_e_ij = af.moddims(u_e_ij, d0 = params.N_LGL, d1 = params.N_LGL,
                            d2 = advec_var.elements.shape[0], d3 = shape_u[2])

        u_edge = af.reorder(u_e_ij[-1, :, :], d0 = 1, d1 = 2, d2 = 3, d3 = 0)

        return u_edge



def u_at_edge_element_wise(u_e_ij, edge_id, element_tags, advec_var,
                           unshared_edge_value = 0):
    '''
    Finds the :math:`u` at given edge id for :math:`u_{eij}`.
    
    [Tested]
    [TODO] 1. Documentation; 2. Unit tests
    
    Parameters
    ----------
    u_e_ij : af.Array [N_LGL^2 N_elements 1 1]
             The u_e_ij array. It can be replaced by the ``x_e_ij`` or
             ``y_e_ij`` array to get the :math:`x` or :math:`y` coordinates
             at the edges.

    edge_id : int
              Edge id.
              To find what each edge id represents, see this
              :py:meth:`dg_maxwell.msh_parser.edge_location`.

    element_tags: af.Array [N 1 1 1] dtype = af.Dtype.u32
                  The element tags for which the ``u_edge`` has to be found.

    advec_var : global_variables.advec_variables
    
    unshared_edge_value : float
                          :math:`u` value to put on the unshared edge.

    Returns
    -------
    u_edge : af.Array [N_LGL N_elements 1 1]
             The :math:`u` at the required edge.
    '''
    u_edge = None
    
    if edge_id == 0:

        u_e_ij = af.reorder(u_e_ij, d0 = 0, d1 = 2, d2 = 1)
        u_e_ij = af.moddims(u_e_ij, d0 = params.N_LGL, d1 = params.N_LGL,
                            d2 = advec_var.elements.shape[0])

        u_edge = af.reorder(u_e_ij[:, 0, element_tags], d0 = 0, d1 = 2, d2 = 1)


    # Get bottom edge of all the elements

    if edge_id == 1:
        
        u_e_ij = af.reorder(u_e_ij, d0 = 0, d1 = 2, d2 = 1)
        u_e_ij = af.moddims(u_e_ij, d0 = params.N_LGL, d1 = params.N_LGL,
                            d2 = advec_var.elements.shape[0])

        u_edge = af.reorder(u_e_ij[0, :, element_tags], d0 = 1, d1 = 2, d2 = 0)

    # Get right edge of all the elements

    if edge_id == 2:

        u_e_ij = af.reorder(u_e_ij, d0 = 0, d1 = 2, d2 = 1)
        u_e_ij = af.moddims(u_e_ij, d0 = params.N_LGL, d1 = params.N_LGL,
                            d2 = advec_var.elements.shape[0])

        u_edge = af.reorder(u_e_ij[:, -1, element_tags], d0 = 0, d1 = 2, d2 = 1)

    # Get top edge of all the elements

    if edge_id == 3:

        u_e_ij = af.reorder(u_e_ij, d0 = 0, d1 = 2, d2 = 1)
        u_e_ij = af.moddims(u_e_ij, d0 = params.N_LGL, d1 = params.N_LGL,
                            d2 = advec_var.elements.shape[0])

        u_edge = af.reorder(u_e_ij[-1, :, element_tags], d0 = 1, d1 = 2, d2 = 0)

    select_unshared_edges_0 = af.transpose(
        advec_var.interelement_relations[:, edge_id] != -1)
    
    select_unshared_edges_1 = af.transpose(
        advec_var.interelement_relations[:, edge_id] == -1)
    
    u_edge = af.broadcast(utils.multiply, select_unshared_edges_0, u_edge)
    u_edge = af.broadcast(utils.add,
                          select_unshared_edges_1 * unshared_edge_value,
                          u_edge)

    return u_edge



def u_at_other_element_edge(u_edge_vec, element_edge_id,
                            advec_var, unshared_edge_value = 0.):
    '''
    When a ``element_edge_id`` edge is shared between two elements, this
    function finds the edge value of the other element at the shared edge.
    
    Parameters
    ----------
    u_edge_vec : af.Array [N_LGL N_elements 4 1]
                 ``u_edge`` at all the edges for all the elements.
    
    element_edge_id : int
                      The edge id of the current element at which the
                      ``u_edge`` of the other element is to be found.
    
    advec_var : global_variables.advec_variables     
    
    Returns
    -------
    u_edge_other_element : af.Array [N_LGL N_elements 1 1]
                           The :math:`u` at the required edge.
    '''
    
    other_element_tags = advec_var.interelement_relations[:, element_edge_id]
    
    element_tags = af.range(advec_var.elements.shape[0], dtype = af.Dtype.u32)
    other_element_edge_id = \
        advec_var.interelement_relations[other_element_tags] != \
            af.tile(element_tags,d0 = 1, d1 = 4)
    
    other_element_edge_id = af.cast(af.iszero(other_element_edge_id),
                                    dtype = af.Dtype.u32)

    u_edge_other_element_vec = u_edge_vec[ \
        :, advec_var.interelement_relations[:,element_edge_id], :]

    u_edge_other_element = af.broadcast(utils.multiply,
                                        af.reorder(other_element_edge_id,
                                                   d0 = 2, d1 = 0, d2 = 1),
                                        u_edge_other_element_vec)

    u_edge_other_element = af.sum(u_edge_other_element, dim = 2)

    select_unshared_edges_0 = af.transpose(
        advec_var.interelement_relations[:, element_edge_id] != -1)

    select_unshared_edges_1 = af.transpose(
        advec_var.interelement_relations[:, element_edge_id] == -1)

    u_edge_other_element = af.broadcast(utils.multiply,
                                        select_unshared_edges_0,
                                        u_edge_other_element)
    
    u_edge_other_element = af.broadcast(utils.add,
                                        select_unshared_edges_1 \
                                            * unshared_edge_value,
                                        u_edge_other_element)

    return u_edge_other_element



def lf_flux_all_edges_vectorized(u_e_ij, advec_var):
    '''
    Calculates the Lax Friedrichs flux at all the edges. It can take a vector
    of ``u_e_ij``.
    '''
    shape_u = utils.shape(u_e_ij)

    left_edge_id = 0
    u_left = u_at_edge(u_e_ij,
                       edge_id = left_edge_id,
                       advec_var = advec_var)
    u_left = af.reorder(u_left, d0 = 0, d1 = 1, d2 = 3, d3 = 2)

    # Bottom edge
    bottom_edge_id = 1
    u_bottom = u_at_edge(u_e_ij,
                         edge_id = bottom_edge_id,
                         advec_var = advec_var)

    u_bottom = af.reorder(u_bottom, d0 = 0, d1 = 1, d2 = 3, d3 = 2)

    # Right edge
    right_edge_id = 2
    u_right = u_at_edge(u_e_ij,
                        edge_id = right_edge_id,
                        advec_var = advec_var)

    u_right = af.reorder(u_right, d0 = 0, d1 = 1, d2 = 3, d3 = 2)

    # Top edge
    top_edge_id = 3
    u_top = u_at_edge(u_e_ij,
                      edge_id = top_edge_id,
                      advec_var = advec_var)

    u_top = af.reorder(u_top, d0 = 0, d1 = 1, d2 = 3, d3 = 2)

    # [LOOKS FINE]

    # Create u_edge_vec
    u_edge_vec = af.constant(0., d0 = params.N_LGL,
                             d1 = advec_var.elements.shape[0],
                             d2 = 4, d3 = shape_u[2], dtype = af.Dtype.f64)

    u_edge_vec[:, :, 0, :] = u_left
    u_edge_vec[:, :, 1, :] = u_bottom
    u_edge_vec[:, :, 2, :] = u_right
    u_edge_vec[:, :, 3, :] = u_top

    ## Create 4 arrays to store the u_edge of the other edge sharing element

    # Left edge

    u_left_other_element = u_at_other_element_edge(u_edge_vec = u_edge_vec,
                                                   element_edge_id = left_edge_id,
                                                   advec_var = advec_var)

    # Bottom edge
    u_bottom_other_element = u_at_other_element_edge(u_edge_vec = u_edge_vec,
                                                     element_edge_id = bottom_edge_id,
                                                     advec_var = advec_var)

    # Right edge
    u_right_other_element = u_at_other_element_edge(u_edge_vec = u_edge_vec,
                                                    element_edge_id = right_edge_id,
                                                    advec_var = advec_var)

    # Top edge
    u_top_other_element = u_at_other_element_edge(u_edge_vec = u_edge_vec,
                                                  element_edge_id = top_edge_id,
                                                  advec_var = advec_var)

    # [VALUES NOT TESTED]

    # Find the LF flux for each edge

    # Left edge

    flux_left = wave_equation_2d.F_x(u_left)
    flux_left_other_element = wave_equation_2d.F_x(u_left_other_element)

    lf_flux_left_edge = lax_friedrichs_flux(u_left_other_element,
                                            flux_left_other_element,
                                            u_left, flux_left)

    # Bottom edge

    flux_bottom = wave_equation_2d.F_y(u_bottom)
    flux_bottom_other_element = wave_equation_2d.F_y(u_bottom_other_element)

    lf_flux_bottom_edge = lax_friedrichs_flux(u_bottom_other_element,
                                              flux_bottom_other_element,
                                              u_bottom, flux_bottom)

    # Right edge

    flux_right = wave_equation_2d.F_x(u_right)
    flux_right_other_element = wave_equation_2d.F_x(u_right_other_element)

    lf_flux_right_edge = lax_friedrichs_flux(u_right, flux_right,
                                             u_right_other_element,
                                             flux_right_other_element)

    # Top edge

    flux_top = wave_equation_2d.F_y(u_top)
    flux_top_other_element = wave_equation_2d.F_y(u_top_other_element)

    lf_flux_top_edge = lax_friedrichs_flux(u_top, flux_top,
                                           u_top_other_element,
                                           flux_top_other_element)

    # Store the fluxes in a [N_elements 4 N_LGL 1]

    element_lf_flux = af.constant(0, d0 = params.N_LGL,
                                  d1 = advec_var.elements.shape[0],
                                  d2 = 4, d3 = shape_u[2],
                                  dtype = af.Dtype.f64)

    element_lf_flux[:, :, left_edge_id]   = lf_flux_left_edge
    element_lf_flux[:, :, bottom_edge_id] = lf_flux_bottom_edge
    element_lf_flux[:, :, right_edge_id]  = lf_flux_right_edge
    element_lf_flux[:, :, top_edge_id]    = lf_flux_top_edge

    element_lf_flux = af.reorder(element_lf_flux, d0 = 1, d1 = 2, d2 = 0)

    return element_lf_flux



#def upwind_flux_x(left_state, right_state, c_x):
    #'''
    #'''
    #if c_x > 0:
        #return left_state
    
    #if c_x == 0:
        #return (left_state + right_state) / 2
    
    #if c_x < 0:
        #return right_state
    
    #return



#def upwind_flux_y(bottom_state, top_state, c_y):
    #'''
    #'''
    #if c_y > 0:
        #return bottom_state

    #if c_y == 0:
        #return (bottom_state + top_state) / 2

    #if c_y < 0:
        #return top_state

    #return



def upwind_flux_x(left_state, right_state, c_x):
    '''
    '''
    E_z = 0.5 * (right_state[:, :, :, 0] + left_state[:, :, :, 0] \
        + right_state[:, :, :, 2] - left_state[:, :, :, 2])
    B_y = 0.5 * (right_state[:, :, :, 0] - left_state[:, :, :, 0] \
        + right_state[:, :, :, 2] + left_state[:, :, :, 2])
    B_x = 0.5 * (right_state[:, :, :, 1] + left_state[:, :, :, 1])
    
    flux = af.constant(0.,
                       d0 = left_state.shape[0],
                       d1 = left_state.shape[1],
                       d2 = 1,
                       d3 = left_state.shape[3],
                       dtype = af.Dtype.f64)

    flux[:, :, :, 0] = - B_y
    flux[:, :, :, 2] = - E_z

    return flux



def upwind_flux_y(bottom_state, top_state, c_y):
    '''
    '''
    E_z = 0.5 * (bottom_state[:, :, :, 0] + top_state[:, :, :, 0] \
        + bottom_state[:, :, :, 1] - top_state[:, :, :, 1])
    B_y = 0.5 * (bottom_state[:, :, :, 2] + top_state[:, :, :, 2])
    B_x = 0.5 * (bottom_state[:, :, :, 0] - top_state[:, :, :, 0] \
        + bottom_state[:, :, :, 1] + top_state[:, :, :, 1])

    shape_bottom_state = utils.shape(bottom_state)
    
    flux = af.constant(0.,
                       d0 = bottom_state.shape[0],
                       d1 = bottom_state.shape[1],
                       d2 = 1,
                       d3 = bottom_state.shape[3],
                       dtype = af.Dtype.f64)

    flux[:, :, :, 0] = B_x
    flux[:, :, :, 1] = E_z

    return flux



def flux_all_edges_upwind_scheme(u_e_ij, advec_var):
    '''
    '''
    shape_u = utils.shape(u_e_ij)

    left_edge_id = 0
    u_left = u_at_edge(u_e_ij,
                       edge_id = left_edge_id,
                       advec_var = advec_var)
    u_left = af.reorder(u_left, d0 = 0, d1 = 1, d2 = 3, d3 = 2)

    # Bottom edge
    bottom_edge_id = 1
    u_bottom = u_at_edge(u_e_ij,
                         edge_id = bottom_edge_id,
                         advec_var = advec_var)

    u_bottom = af.reorder(u_bottom, d0 = 0, d1 = 1, d2 = 3, d3 = 2)

    # Right edge
    right_edge_id = 2
    u_right = u_at_edge(u_e_ij,
                        edge_id = right_edge_id,
                        advec_var = advec_var)

    u_right = af.reorder(u_right, d0 = 0, d1 = 1, d2 = 3, d3 = 2)

    # Top edge
    top_edge_id = 3
    u_top = u_at_edge(u_e_ij,
                      edge_id = top_edge_id,
                      advec_var = advec_var)

    u_top = af.reorder(u_top, d0 = 0, d1 = 1, d2 = 3, d3 = 2)

    # [LOOKS FINE]

    # Create u_edge_vec
    u_edge_vec = af.constant(0., d0 = params.N_LGL,
                             d1 = advec_var.elements.shape[0],
                             d2 = 4, d3 = shape_u[2], dtype = af.Dtype.f64)

    u_edge_vec[:, :, 0, :] = u_left
    u_edge_vec[:, :, 1, :] = u_bottom
    u_edge_vec[:, :, 2, :] = u_right
    u_edge_vec[:, :, 3, :] = u_top

    ## Create 4 arrays to store the u_edge of the other edge sharing element

    # Left edge

    u_left_other_element = u_at_other_element_edge(u_edge_vec = u_edge_vec,
                                                   element_edge_id = left_edge_id,
                                                   advec_var = advec_var)

    # Bottom edge
    u_bottom_other_element = u_at_other_element_edge(u_edge_vec = u_edge_vec,
                                                     element_edge_id = bottom_edge_id,
                                                     advec_var = advec_var)

    # Right edge
    u_right_other_element = u_at_other_element_edge(u_edge_vec = u_edge_vec,
                                                    element_edge_id = right_edge_id,
                                                    advec_var = advec_var)

    # Top edge
    u_top_other_element = u_at_other_element_edge(u_edge_vec = u_edge_vec,
                                                  element_edge_id = top_edge_id,
                                                  advec_var = advec_var)

    # [VALUES NOT TESTED]

    # Find the LF flux for each edge

    # Left edge

    u_at_left_edge = upwind_flux_x(u_left_other_element, u_left, params.c_x)
    flux_left_edge = u_at_left_edge
    # [NOTE]: Comment the next line when evolving Maxwell's equations
    #flux_left_edge = wave_equation_2d.F_x(u_at_left_edge)

    u_at_bottom_edge = upwind_flux_y(u_bottom_other_element, u_bottom, params.c_y)
    flux_bottom_edge = u_at_bottom_edge
    # [NOTE]: Comment the next line when evolving Maxwell's equations
    #flux_bottom_edge = wave_equation_2d.F_y(u_at_bottom_edge)

    u_at_right_edge = upwind_flux_x(u_right, u_right_other_element, params.c_x)
    flux_right_edge = u_at_right_edge
    # [NOTE]: Comment the next line when evolving Maxwell's equations
    #flux_right_edge = wave_equation_2d.F_x(u_at_right_edge)

    u_at_top_edge = upwind_flux_y(u_top, u_top_other_element, params.c_y)
    flux_top_edge = u_at_top_edge
    # [NOTE]: Comment the next line when evolving Maxwell's equations
    #flux_top_edge = wave_equation_2d.F_y(u_at_top_edge)

    # Store the fluxes in a [N_elements 4 N_LGL 1]
    element_lf_flux = af.constant(0, d0 = params.N_LGL,
                                  d1 = advec_var.elements.shape[0],
                                  d2 = 4, d3 = shape_u[2],
                                  dtype = af.Dtype.f64)

    element_lf_flux[:, :, left_edge_id]   = flux_left_edge
    element_lf_flux[:, :, bottom_edge_id] = flux_bottom_edge
    element_lf_flux[:, :, right_edge_id]  = flux_right_edge
    element_lf_flux[:, :, top_edge_id]    = flux_top_edge

    element_lf_flux = af.reorder(element_lf_flux, d0 = 1, d1 = 2, d2 = 0)

    return element_lf_flux



def surface_term_vectorized(u, advec_var):
    '''
    Calculates the surface term for 2D wave equation :math:`x-y` formulation.
    This function will work for arbitrary non-contiguous meshes on a condition
    that the elements in the meshes are square of dimention :math:`0.2` units.
    See link for :math:`x-y` formulation surface term see `2d_advection.pdf`_
    
    .. _2d_advection.pdf: https://goo.gl/jBmgh6
    
    .. note::
       See date 2017-12-18 in the Research Notebook 2 for the implementation logs.

    Parameters
    ----------
    u : af.Array [N_LGL N_elements M 1]
        ``u_e_ij`` for which the surface term is to be calculated.
        This function can calculate the surface term for :math:`M`
        ``u``.
        simultaneously.

    advec_var : :py:meth:`dg_maxwell.global_variables.advection_variables`
    
    Returns
    -------
    surface_term : af.Array [N_LGL*N_LGL N_elements M 1]
                   Surface term calculated for :math:`M` ``u``.
    '''
    dx_dxi  = 0.1
    dy_deta = 0.1

    shape_u = utils.shape(u)

    #element_lf_flux = lf_flux_all_edges_vectorized(u, advec_var)
    
    element_lf_flux = flux_all_edges_upwind_scheme(u, advec_var)
    
    # 1. Find L_p(1) and L_p(-1)
    Lp = advec_var.lagrange_coeffs
    Lq = advec_var.lagrange_coeffs

    Lp_1           = utils.polyval_1d(Lp, af.constant(1, d0 = 1,
                                                      dtype = af.Dtype.f64))
    Lp_1_slow_tile = af.moddims(af.transpose(af.tile(Lp_1, d0 = 1,
                                                     d1 = params.N_LGL,
                                                     d2 = params.N_LGL)),
                                d0 = params.N_LGL * params.N_LGL,
                                d1 = params.N_LGL)

    Lp_minus_1           = utils.polyval_1d(Lp, af.constant(-1, d0 = 1,
                                                            dtype = af.Dtype.f64))
    Lp_minus_1_slow_tile = af.moddims(af.transpose(af.tile(Lp_minus_1, d0 = 1,
                                                           d1 = params.N_LGL,
                                                           d2 = params.N_LGL)),
                                      d0 = params.N_LGL * params.N_LGL,
                                      d1 = params.N_LGL)

    Lq_1 = Lp_1.copy()
    Lq_1_quick_tile = af.tile(Lq_1, d0 = params.N_LGL, d1 = params.N_LGL)

    Lq_minus_1 = Lp_minus_1.copy()
    Lq_minus_1_quick_tile = af.tile(Lq_minus_1,
                                    d0 = params.N_LGL,
                                    d1 = params.N_LGL)

    # 2. Find Lp(xi) and Lq(eta) [N_LGL * N_LGL, N_LGL]

    Lp_xi_i = utils.polyval_1d(Lp, advec_var.xi_LGL)
    Lp_xi_i_slow_tile = af.tile(af.reorder(Lp_xi_i, d0 = 2, d1 = 0, d2 = 1),
                                d0 = params.N_LGL)
    Lp_xi_i_slow_tile = af.moddims(Lp_xi_i_slow_tile,
                                   d0 = params.N_LGL * params.N_LGL,
                                   d1 = params.N_LGL, d2 = 1)


    Lq_eta_j = utils.polyval_1d(Lq, advec_var.eta_LGL)
    Lq_eta_j_quick_tile = af.tile(Lq_eta_j, d0 = params.N_LGL)

    dx_dxi_tile = af.constant(dx_dxi,
                              d0 = params.N_LGL * params.N_LGL,
                              d1 = params.N_LGL)
    dy_deta_tile = af.constant(dy_deta,
                               d0 = params.N_LGL * params.N_LGL,
                               d1 = params.N_LGL)

    F_xi_minus_1_eta_j = af.transpose(
        af.tile(af.reorder(element_lf_flux[:, 0],
                           d0 = 2, d1 = 1, d2 = 0),
                d0 = 1, d1 = params.N_LGL * params.N_LGL))

    F_xi_i_eta_minus_1 = af.transpose(
        af.tile(af.reorder(element_lf_flux[:, 1],
                           d0 = 2, d1 = 1, d2 = 0),
        d0 = 1, d1 = params.N_LGL * params.N_LGL))

    F_xi_1_eta_j       = af.transpose(
        af.tile(af.reorder(element_lf_flux[:, 2],
                           d0 = 2, d1 = 1, d2 = 0),
        d0 = 1, d1 = params.N_LGL * params.N_LGL))

    F_xi_i_eta_1       = af.transpose(
        af.tile(af.reorder(element_lf_flux[:, 3],
                           d0 = 2, d1 = 1, d2 = 0),
        d0 = 1, d1 = params.N_LGL * params.N_LGL))

    # 5. Calculate the surface term intergal for the left edge

    integrand_left_edge = af.broadcast(utils.multiply, F_xi_minus_1_eta_j,
                                       Lp_minus_1_slow_tile \
                                     * Lq_eta_j_quick_tile  \
                                     * dy_deta_tile)

    integrand_left_edge_merge_elements = \
        af.transpose(af.moddims(af.transpose(integrand_left_edge),
                                d0 = integrand_left_edge.shape[1],
                                d1 = integrand_left_edge.shape[0] \
                                   * integrand_left_edge.shape[2] \
                                   * shape_u[2],
                                d2 = 1))

    integrand_left_edge_merge_elements = lagrange.lagrange_interpolation(
        integrand_left_edge_merge_elements, advec_var)
    integral_left_edge_merge_elements  = utils.integrate_1d(
        integrand_left_edge_merge_elements,
        order = params.N_LGL + 1,
        scheme = 'gauss')

    integral_left_edge = af.moddims(integral_left_edge_merge_elements,
                                    d0 = integrand_left_edge.shape[0],
                                    d1 = integrand_left_edge.shape[2],
                                    d2 = shape_u[2])

    # 6. Calculate the surface term intergal for the bottom edge

    integrand_bottom_edge = af.broadcast(utils.multiply, F_xi_i_eta_minus_1,
                                         Lp_xi_i_slow_tile \
                                       * Lq_minus_1_quick_tile \
                                       * dx_dxi_tile)
    integrand_bottom_edge_merge_elements = af.transpose(
        af.moddims(af.transpose(integrand_bottom_edge),
                   d0 = integrand_bottom_edge.shape[1],
                   d1 = integrand_bottom_edge.shape[0] \
                      * integrand_bottom_edge.shape[2] \
                      * shape_u[2],
                   d2 = 1))

    integrand_bottom_edge_merge_elements = lagrange.lagrange_interpolation(
        integrand_bottom_edge_merge_elements, advec_var)
    integral_bottom_edge_merge_elements  = utils.integrate_1d(
        integrand_bottom_edge_merge_elements, order = params.N_LGL + 1,
        scheme = 'gauss')
    integral_bottom_edge = af.moddims(integral_bottom_edge_merge_elements,
                                      d0 = integrand_bottom_edge.shape[0],
                                      d1 = integrand_bottom_edge.shape[2],
                                      d2 = shape_u[2])


    # 7. Calculate the surface term intergal for the right edge

    integrand_right_edge = af.broadcast(utils.multiply, F_xi_1_eta_j,
                                        Lp_1_slow_tile \
                                      * Lq_eta_j_quick_tile \
                                      * dy_deta_tile)
    integrand_right_edge_merge_elements = af.transpose(
        af.moddims(af.transpose(integrand_right_edge),
                   d0 = integrand_right_edge.shape[1],
                   d1 = integrand_right_edge.shape[0] \
                      * integrand_right_edge.shape[2] \
                      * shape_u[2],
                   d2 = 1))

    integrand_right_edge_merge_elements = lagrange.lagrange_interpolation(
        integrand_right_edge_merge_elements, advec_var)
    integral_right_edge_merge_elements  = utils.integrate_1d(
        integrand_right_edge_merge_elements, order = params.N_LGL + 1,
        scheme = 'gauss')
    integral_right_edge = af.moddims(integral_right_edge_merge_elements,
                                     d0 = integrand_right_edge.shape[0],
                                     d1 = integrand_right_edge.shape[2],
                                     d2 = shape_u[2])


    # 8. Calculate the surface term intergal for the top edge

    integrand_top_edge = af.broadcast(utils.multiply, F_xi_i_eta_1,
                                      Lp_xi_i_slow_tile \
                                    * Lq_1_quick_tile   \
                                    * dx_dxi_tile)
    integrand_top_edge_merge_elements = af.transpose(
        af.moddims(af.transpose(integrand_top_edge),
                   d0 = integrand_top_edge.shape[1],
                   d1 = integrand_top_edge.shape[0] \
                      * integrand_top_edge.shape[2] \
                      * shape_u[2],
                   d2 = 1))
    integrand_top_edge_merge_elements = lagrange.lagrange_interpolation(
        integrand_top_edge_merge_elements, advec_var)
    integral_top_edge_merge_elements  = utils.integrate_1d(
        integrand_top_edge_merge_elements, order = params.N_LGL + 1,
        scheme = 'gauss')
    integral_top_edge = af.moddims(integral_top_edge_merge_elements,
                                   d0 = integrand_top_edge.shape[0],
                                   d1 = integrand_top_edge.shape[2],
                                   d2 = shape_u[2])

    surface_term = - integral_left_edge   \
                   - integral_bottom_edge \
                   + integral_right_edge  \
                   + integral_top_edge

    return surface_term


def b_vector(u, advec_var):
    '''
    '''
    b = volume_integral(u, advec_var) \
      - surface_term_vectorized(u, advec_var)

    return b



def RK4_timestepping(A_inverse, u, delta_t, gv):
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

    k1 = utils.matmul_3D(A_inverse, b_vector(u, gv))
    k2 = utils.matmul_3D(A_inverse, b_vector(u + k1 * delta_t / 2, gv))
    k3 = utils.matmul_3D(A_inverse, b_vector(u + k2 * delta_t / 2, gv))
    k4 = utils.matmul_3D(A_inverse, b_vector(u + k3 * delta_t, gv))

    delta_u = delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return delta_u



def time_evolution(u_init, gv):
    '''
    '''
    # Creating a folder to store hdf5 files. If it doesn't exist.
    results_directory = 'results/2d_hdf5_%02d' %(int(params.N_LGL))
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    u         = u_init
    shape_u   = utils.shape(u)
    delta_t   = gv.delta_t_2d
    time      = gv.time_2d

    gauss_points    = gv.gauss_points
    gauss_weights   = gv.gauss_weights
    dLp_Lq          = gv.dLp_Lq
    dLq_Lp          = gv.dLq_Lp
    xi_LGL          = gv.xi_LGL
    lagrange_coeffs = gv.lagrange_coeffs
    Li_Lj_coeffs    = gv.Li_Lj_coeffs
    lobatto_weights = gv.lobatto_weights_quadrature

    A_inverse = af.np_to_af_array(np.linalg.inv(np.array(A_matrix(gv))))
    A_inverse = af.tile(A_inverse, d0 = 1, d1 = 1, d2 = shape_u[2])

    for i in trange(time.shape[0]):
        L1_norm = af.mean(af.abs(u_init - u))

        if (L1_norm >= 100):
            break
        if (i % 1) == 0:
            h5file = h5py.File('results/2d_hdf5_%02d/dump_timestep_%06d' \
                %(int(params.N_LGL), int(i)) + '.hdf5', 'w')
            dset   = h5file.create_dataset('u_i', data = u, dtype = 'd')

            dset[:, :] = u[:, :]
            h5file.close()


        u += +RK4_timestepping(A_inverse, u, delta_t, gv)

    return L1_norm



def u_analytical(t_n, gv):
    '''
    '''
    time = gv.delta_t_2d * t_n
    u_analytical_t_n = af.sin(2 * np.pi * (gv.x_e_ij - params.c_x * time) +
                              4 * np.pi * (gv.y_e_ij - params.c_y * time))

    return u_analytical_t_n
