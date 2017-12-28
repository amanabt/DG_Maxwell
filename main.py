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

sigma = 0.4
E_z_init = np.e**(- (gv.x_e_ij**2) / sigma**2)
B_x_init = af.constant(0., d0 = gv.x_e_ij.shape[0],
                       d1 = gv.x_e_ij.shape[1],
                       dtype = af.Dtype.f64)
B_y_init = np.e**(- (gv.x_e_ij**2) / sigma**2)

u_init = af.join(dim    = 2,
                 first  = E_z_init,
                 second = B_x_init,
                 third  = B_y_init)
#u_init = E_z_init #[NOTE] Comment this when running the Maxwell's equations

advection_2d_arbit_mesh.time_evolution(u_init, gv)


#change_parameters(5)
#print(advection_2d.time_evolution())
#
#L1_norm = np.zeros([5])
#for LGL in range(3, 8):
#    print(LGL)
#    change_parameters(LGL)
#    L1_norm[LGL - 3] = (advection_2d.time_evolution())
#    print(L1_norm[LGL - 3])
#
#print(L1_norm)
#
#L1_norm = np.array([8.20284941e-02, 1.05582246e-02, 9.12125969e-04, 1.26001632e-04, 8.97007162e-06, 1.0576058855881385e-06])
#LGL = (np.arange(6) + 3).astype(float)
#normalization = 8.20284941e-02 / (3 ** (-3 * 0.85))
#pl.loglog(LGL, L1_norm, marker='o', label='L1 norm')
#pl.loglog(LGL, normalization * LGL ** (-0.85 * LGL), color='black', linestyle='--', label='$N_{LGL}^{-0.85 N_{LGL}}$')
#pl.title('L1 norm v/s $N_{LGL}$')
#pl.legend(loc='best')
#pl.xlabel('$N_{LGL}$ points')
#pl.ylabel('L1 norm of error')
#pl.show()
