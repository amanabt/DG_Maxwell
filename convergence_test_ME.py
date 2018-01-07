#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
from matplotlib import pyplot as pl
from tqdm import trange
import numpy as np

from dg_maxwell import params
from dg_maxwell import advection_2d_arbit_mesh
from dg_maxwell import utils
from dg_maxwell import global_variables
from dg_maxwell import msh_parser

af.set_backend(params.backend)
af.set_device(params.device)

error = []

for N_LGL in trange(4, 25):
    print(N_LGL)
    params.N_LGL = N_LGL
    params.N_quad = N_LGL

    advec_var = global_variables.advection_variables(params.N_LGL, params.N_quad,
                                                     params.x_nodes, params.N_Elements,
                                                     params.c, params.total_time, params.wave,
                                                     params.c_x, params.c_y, params.courant,
                                                     params.mesh_file, params.total_time_2d)

    edge_reorded_mesh = msh_parser.rearrange_element_edges(advec_var.elements, advec_var)
    advec_var.reassign_2d_elements(edge_reorded_mesh)

    print('Delta t 2D', advec_var.delta_t_2d)

    # 1. Sin initial conditions

    E_z_init = af.sin(2 * np.pi * advec_var.x_e_ij) \
                + af.cos(2 * np.pi * advec_var.y_e_ij)
    B_x_init = af.cos(2 * np.pi * advec_var.y_e_ij)
    B_y_init = af.sin(2 * np.pi * advec_var.x_e_ij)

    u_init = af.join(dim    = 2,
                     first  = E_z_init,
                     second = B_x_init,
                     third  = B_y_init)

    error.append(advection_2d_arbit_mesh.time_evolution(u_init, advec_var))
    print(error[-1])
    
error = np.array(error)
print('Error', error)
np.savetxt('convergence_ME_2d_error.csv', error, delimiter=',')
