from netCDF4 import Dataset
import numpy as np

e = 4.0
nz = 72

d = Dataset('INPUT/coord.nc', 'w')
d.createDimension('Layer', nz)
d.createVariable('Layer', 'f8', ('Layer',))[:] = 1030 + np.tanh(np.linspace(0, e, nz)) / np.tanh(e) * 7.7
d.close()
