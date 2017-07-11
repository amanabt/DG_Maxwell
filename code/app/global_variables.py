import numpy as np
import arrayfire as af
from scipy import special as sp
from app import lagrange


LGL_list = [ \
[-1.0,1.0],                                                               \
[-1.0,0.0,1.0],                                                           \
[-1.0,-0.4472135955,0.4472135955,1.0],                                    \
[-1.0,-0.654653670708,0.0,0.654653670708,1.0],                            \
[-1.0,-0.765055323929,-0.285231516481,0.285231516481,0.765055323929,1.0], \
[-1.0,-0.830223896279,-0.468848793471,0.0,0.468848793471,0.830223896279,  \
1.0],                                                                     \
[-1.0,-0.87174014851,-0.591700181433,-0.209299217902,0.209299217902,      \
0.591700181433,0.87174014851,1.0],                                        \
[-1.0,-0.899757995411,-0.677186279511,-0.363117463826,0.0,0.363117463826, \
0.677186279511,0.899757995411,1.0],                                       \
[-1.0,-0.919533908167,-0.738773865105,-0.47792494981,-0.165278957666,     \
0.165278957666,0.47792494981,0.738773865106,0.919533908166,1.0],          \
[-1.0,-0.934001430408,-0.784483473663,-0.565235326996,-0.295758135587,    \
0.0,0.295758135587,0.565235326996,0.784483473663,0.934001430408,1.0],     \
[-1.0,-0.944899272223,-0.819279321644,-0.632876153032,-0.399530940965,    \
-0.136552932855,0.136552932855,0.399530940965,0.632876153032,             \
0.819279321644,0.944899272223,1.0],                                       \
[-1.0,-0.953309846642,-0.846347564652,-0.686188469082,-0.482909821091,    \
-0.249286930106,0.0,0.249286930106,0.482909821091,0.686188469082,         \
0.846347564652,0.953309846642,1.0],                                       \
[-0.999999999996,-0.959935045274,-0.867801053826,-0.728868599093,         \
-0.550639402928,-0.342724013343,-0.116331868884,0.116331868884,           \
0.342724013343,0.550639402929,0.728868599091,0.86780105383,               \
0.959935045267,1.0],                                                      \
[-0.999999999996,-0.965245926511,-0.885082044219,-0.763519689953,         \
-0.60625320547,-0.420638054714,-0.215353955364,0.0,0.215353955364,        \
0.420638054714,0.60625320547,0.763519689952,0.885082044223,               \
0.965245926503,1.0],                                                      \
[-0.999999999984,-0.9695680463,-0.899200533072,-0.792008291871,           \
-0.65238870288,-0.486059421887,-0.299830468901,-0.101326273522,           \
0.101326273522,0.299830468901,0.486059421887,0.652388702882,              \
0.792008291863,0.899200533092,0.969568046272,0.999999999999]]


for idx in np.arange(len(LGL_list)):
	LGL_list[idx] = np.array(LGL_list[idx], dtype = np.float64)
	LGL_list[idx] = af.interop.np_to_af_array(LGL_list[idx])

x_nodes     = af.interop.np_to_af_array(np.array([[-2., 2.]]))
N_LGL       = 16
xi_LGL      = None
lBasisArray = None

def populateGlobalVariables(N = 16):
	'''
	Initialize the global variables.

	Parameters
	----------
	N : int
		Number of LGL points.
	Declares the number and the value of
	'''
	global N_LGL
	global xi_LGL
	global lBasisArray

	N_LGL       = N
	xi_LGL      = lagrange.LGL_points(N_LGL)
	lBasisArray = af.interop.np_to_af_array( \
		lagrange.lagrange_basis_coeffs(xi_LGL))
	
	return


def lobatto_weight_function(n, x):
	'''
	Calculates and returns the weight function for an index n
	and points x
	
	:math::
		w_{n} = \frac{2 P(x)^2}{n (n - 1)}, 
		Where P(x) is (n - 1)^th index.
	
	Parameters
	----------
	n : int
		Index for which lobatto weight function

	x : arrayfire.Array
		1D array of points where weight function is to be calculated.

	.. lobatto weight function - https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss.E2.80.93Lobatto_rules
	
	Returns
	-------
	An array of lobatto weight functions for the given x points and index.
	
	'''
	P = sp.legendre(n - 1)
	
	return (2 / (n * (n - 1)) / (P(x))**2)
