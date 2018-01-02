#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
import numpy as np
import os
import numpy as np
import arrayfire as af

from matplotlib import pyplot as pl
from tqdm import trange

from dg_maxwell import wave_equation
from dg_maxwell import msh_parser
from dg_maxwell import wave_equation_2d
from dg_maxwell.tests import test_waveEqn
from dg_maxwell import isoparam
from dg_maxwell import lagrange
from dg_maxwell import params
from dg_maxwell import advection_2d
from dg_maxwell import advection_2d_arbit_mesh
from dg_maxwell import utils
from dg_maxwell import global_variables

af.set_backend(params.backend)
af.set_device(params.device)

#print(af.mean(af.abs(advection_2d.u_analytical(0) - params.u_e_ij)))

gv = global_variables.advection_variables(params.N_LGL, params.N_quad,
                                          params.x_nodes, params.N_Elements,
                                          params.c, params.total_time, params.wave,
                                          params.c_x, params.c_y, params.courant,
                                          params.mesh_file, params.total_time_2d)

edge_reorded_mesh = msh_parser.rearrange_element_edges(gv.elements, gv)
gv.reassign_2d_elements(edge_reorded_mesh)

print('Delta t 2D', gv.delta_t_2d)

## 1. Gaussian initial conditions
#sigma = 0.4
#E_z_init = np.e**(- (gv.x_e_ij**2 + gv.y_e_ij**2) / sigma**2)
#B_x_init = np.e**(- (gv.y_e_ij**2) / sigma**2)
#B_x_init = af.constant(0., d0 = gv.x_e_ij.shape[0],
                       #d1 = gv.x_e_ij.shape[1],
                       #dtype = af.Dtype.f64)
#B_y_init = np.e**(- (gv.x_e_ij**2) / sigma**2)
#B_y_init = af.constant(0., d0 = gv.x_e_ij.shape[0],
                       #d1 = gv.x_e_ij.shape[1],
                       #dtype = af.Dtype.f64)

# 1. Sin initial conditions

E_z_init = af.sin(2 * np.pi * gv.x_e_ij) + af.cos(2 * np.pi * gv.y_e_ij)
B_x_init = af.cos(2 * np.pi * gv.y_e_ij)
#B_x_init = af.constant(0., d0 = gv.x_e_ij.shape[0],
                       #d1 = gv.x_e_ij.shape[1],
                       #dtype = af.Dtype.f64)
B_y_init = af.sin(2 * np.pi * gv.x_e_ij)
#B_y_init = af.constant(0., d0 = gv.x_e_ij.shape[0],
                       #d1 = gv.x_e_ij.shape[1],
                       #dtype = af.Dtype.f64)

u_init = af.join(dim    = 2,
                 first  = E_z_init,
                 second = B_x_init,
                 third  = B_y_init)
#u_init = E_z_init #[NOTE] Comment this when running the Maxwell's equations

advection_2d_arbit_mesh.time_evolution(u_init, gv)
