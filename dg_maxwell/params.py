#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import arrayfire as af

backend = 'opencl'
device = 0

af.set_backend(backend)
af.set_device(device)

af.info()

# The domain of the function.
x_nodes    = af.np_to_af_array(np.array([-1., 1.]))

# The number of LGL points into which an element is split.
N_LGL      = 8

# Number of elements the domain is to be divided into.
N_Elements = 10

# The scheme to be used for integration. Values are either
# 'gauss_quadrature' or 'lobatto_quadrature'
scheme     = 'gauss_quadrature'

# The scheme to integrate the volume integral flux
volume_integral_scheme = 'lobatto_quadrature'

# The number quadrature points to be used for integration.
N_quad     = 8

# Wave speed.
c          = 1

# The total time for which the wave is to be evolved by the simulation.
total_time = 2.0

# The c_lax to be used in the Lax-Friedrichs flux.
c_lax      = c

# The wave to be advected is either a sin or a Gaussian wave.
# This parameter can take values 'sin' or 'gaussian'.
wave = 'sin'

c_x = 1.


# The parameters below are for 2D advection
# -----------------------------------------


########################################################################
#######################2D Wave Equation#################################
########################################################################

c_x = 1.
c_y = 0.

c_lax_2d = c_x

courant = 0.1

mesh_file = 'examples/read_and_plot_mesh/mesh/square_10_10.msh'
#mesh_file = 'examples/read_and_plot_mesh/mesh/square_contiguous_4_4.msh'
#mesh_file = 'examples/read_and_plot_mesh/mesh/square_mesh_4_parts.msh'
#mesh_file = 'examples/read_and_plot_mesh/mesh/particle_in_rectangle.msh'
#mesh_file  = 'examples/read_and_plot_mesh/mesh/square_1_3.msh'

total_time_2d = 4.01

volume_integrand_scheme_2d = 'Lobatto'

#################################################################
## Periodic boundary conditions for 10x10 mesh
#################################################################
vertical_boundary_elements_pbc   = np.zeros([10, 2], dtype = np.int64)
horizontal_boundary_elements_pbc = np.zeros([10, 2], dtype = np.int64)

for idx in np.arange(vertical_boundary_elements_pbc.shape[0]):
    vertical_boundary_elements_pbc[idx] = np.array([idx * 10, idx * 10 + 9])
print(vertical_boundary_elements_pbc)

for idx in np.arange(horizontal_boundary_elements_pbc.shape[0]):
    horizontal_boundary_elements_pbc[idx] = np.array([idx, idx + 90])
print(horizontal_boundary_elements_pbc)


#################################################################
## Periodic boundary conditions for particle in a rectangle mesh
#################################################################

##                                left   right
#vertical_boundary_elements_pbc = [[0 ,  184],
                                  #[10,  194],
                                  #[20,  204],
                                  #[30,  214],
                                  #[40,  224],
                                  #[50,  174],
                                  #[60,  164],
                                  #[70,  154],
                                  #[80,  144],
                                  #[90,  134],
                                  #[100, 124]]
#vertical_boundary_elements_pbc = np.array(vertical_boundary_elements_pbc)

##                                    Top  Bottom
#horizontal_boundary_elements_pbc = [[  0, 100],
                                    #[  1, 101],
                                    #[  2, 102],
                                    #[  3, 103],
                                    #[  4, 104],
                                    #[  5, 105],
                                    #[  6, 106],
                                    #[  7, 107],
                                    #[  8, 108],
                                    #[  9, 109],
                                    #[225, 110],
                                    #[175, 115],
                                    #[176, 116],
                                    #[177, 117],
                                    #[178, 118],
                                    #[179, 119],
                                    #[180, 120],
                                    #[181, 121],
                                    #[182, 122],
                                    #[183, 123],
                                    #[184, 124]]
#horizontal_boundary_elements_pbc = np.array(horizontal_boundary_elements_pbc)


######################################################################
### Periodic boundary conditions for square 10x10 non-contiguous mesh
######################################################################

#vertical_boundary_elements_pbc = [[ 0,  99],
                                  #[ 5,  94],
                                  #[10,  89],
                                  #[15,  84],
                                  #[20,  79],
                                  #[25,  74],
                                  #[30,  73],
                                  #[35,  72],
                                  #[40,  71],
                                  #[45,  70]]

#vertical_boundary_elements_pbc = np.array(vertical_boundary_elements_pbc)


#horizontal_boundary_elements_pbc = [[ 0,  45],
                                    #[ 1,  46],
                                    #[ 2,  47],
                                    #[ 3,  48],
                                    #[ 4,  49],
                                    #[95,  50],
                                    #[96,  55],
                                    #[97,  60],
                                    #[98,  65],
                                    #[99,  70]]

#horizontal_boundary_elements_pbc = np.array(horizontal_boundary_elements_pbc)

#####################################################################
## Periodic boundary conditions for square 4x4 non-contiguous mesh
#####################################################################

#vertical_boundary_elements_pbc = [[ 0,  15],
                                  #[ 2,  13],
                                  #[ 4,  11],
                                  #[ 6,  10]]

#vertical_boundary_elements_pbc = np.array(vertical_boundary_elements_pbc)


#horizontal_boundary_elements_pbc = [[ 0,   6],
                                    #[ 1,   7],
                                    #[14,   8],
                                    #[15,  10]]

#horizontal_boundary_elements_pbc = np.array(horizontal_boundary_elements_pbc)


#####################################################################
## Periodic boundary conditions for square 4x4 non-contiguous mesh
#####################################################################

#vertical_boundary_elements_pbc = [[ 0,   2]]

#vertical_boundary_elements_pbc = np.array(vertical_boundary_elements_pbc)


#horizontal_boundary_elements_pbc = [[ 0,   0],
                                    #[ 1,   1],
                                    #[ 2,   2]]

#horizontal_boundary_elements_pbc = np.array(horizontal_boundary_elements_pbc)

