#! /usr/bin/env python3

from os import sys

import arrayfire as af
af.set_backend('opencl')
from app import global_variables as gvar
from app import lagrange
from app import wave_equation

if __name__ == '__main__':
	
	gvar.populateGlobalVariables(4)
	print(wave_equation.A_matrix())
