#!/usr/bin/env python

import numpy as np
from netCDF4 import Dataset

# physical constants
REa = 6.378e6 # radius of earth
D   = 4000   # depth of water

# grid dimensions
nx = 160
ny = 800
nz = 72

# resolution (degrees)
rx = 0.25

# create the latitudinal grid (Mercator projection)
dy = np.zeros(ny)
y  = np.zeros(ny)
dy[ny/2] = rx
y[ny/2]  = rx / 2

for i in range(1, ny/2):
    y[ny/2 + i] = y[ny/2 + i - 1] + dy[ny/2 + i - 1]
    dy[ny/2 + i] = rx * np.cos(y[ny/2 + i] * np.pi / 180)
Yend = y[-1] + dy[-1]/2

# flip onto southern hemisphere
y[:ny/2]  = -y[:ny/2 - 1:-1]
dy[:ny/2] = dy[:ny/2 - 1:-1]

# create the longitudinal grid
# spacing is constant with latitude and longitude
dx = rx * np.ones(nx)
x = np.cumsum(dx) - dx/2

#Y,  X  = np.meshgrid(y, x)
#DY, DX = np.meshgrid(dy, dx)
X,  Y  = np.meshgrid(x, y)
DX, DY = np.meshgrid(dx, dy)

# make land
dc = 59
h = np.minimum(0,               D*(2**(-0.4 * X**2)           * (1 - 0.5 * 2**(-0.003 * np.abs(-Y - dc)**4)) - 0.9) / 0.9)
h = np.maximum(h, np.minimum(0, D*(2**(-0.4 * (X - nx*rx)**2) * (1 - 0.5 * 2**(-0.003 * np.abs(-Y - dc)**4)) - 0.9) / 0.9))

# add southern shelf
shelf_depth = 800
sws = 3 # shelf width scale in degrees
htmp = -D * np.ones(h.shape)
htmp[Y < -Yend + sws] = -shelf_depth * (1 + (Y[Y < -Yend + sws] - (-Yend + sws))**3 / (sws**3))
htmp[(Y < -Yend + 2*sws) & (Y >= -Yend + sws)] = -shelf_depth - (D - shelf_depth)/2 * (Y[(Y < -Yend + 2*sws) & (Y >= -Yend + sws)] - (-Yend + sws))**3 / (sws**3)
htmp[(Y < -Yend + 3*sws) & (Y >= -Yend + 2*sws)] = -D - (D - shelf_depth)/2 * (Y[(Y < -Yend + 3*sws) & (Y >= -Yend + 2*sws)] - (-Yend + 3*sws))**3 / (sws**3)

h = np.minimum(np.maximum(h, htmp), 0)

# shift to the right by two gridpoints
h = np.roll(h, 2, 1)

h[[0,-1],:] = 0

# wind stress
tau_x = 0.03*(np.cos(8*np.pi * Y/180) - 1 + 5.1*np.exp(-((Y + 50) / 20)**2) + 3.2*np.exp(-((Y - 53) / 20)**2))

# SST relaxation
SST = 38 * np.exp(-(np.abs(Y) / 45)**3) - 6

# salinity forcing
Amp = 1e-7 * 0.225
Fs = Amp * (1.2*np.cos(np.pi * Y/60) / np.cos(np.deg2rad(0.2 * Y)) - 3*np.exp(-(Y / (0.15 * 60))**2) + 0.25*np.abs(Y/60))

# calculate surface area of ocean cells
sa = (DY * REa * np.pi/180) * (DX * REa * np.pi/180 * np.cos(Y * np.pi/180))
sa[h >= 0] = 0
sumA = np.sum(sa)

Fs -= np.sum(sa * Fs) / sumA

# create vertical grid
dzmin = 5
dzmax = 98
decay = 512.3207
z = np.empty(nz)
zw = np.empty(nz+1)
dz = np.empty(nz)
z[0] = -dzmin / 2
zw[0] = 0
for i in range(0, nz - 1):
    dz[i] = dzmax - (dzmax - dzmin) / np.cosh(z[i] / decay)**2
    z[i+1] = z[i] - dz[i]
    zw[i+1] = -(z[i] + z[i+1]) / 2
dz[nz - 1] = D - dz.sum()
zw[nz] = D

# write out data
with Dataset('indata.nc', 'w') as d:
    d.createDimension('x', nx)
    d.createDimension('y', ny)
    d.createDimension('z', nz)
    d.createDimension('zt', nz)
    d.createDimension('zw', nz+1)

    d.createVariable('x', 'f8', ('x',))[:] = x
    d.createVariable('y', 'f8', ('y',))[:] = y
    d.createVariable('z', 'f8', ('z',))[:] = z
    d.createVariable('zt', 'f8', ('zt',))[:] = -z
    d.createVariable('zw', 'f8', ('zw',))[:] = zw
    d.createVariable('dz', 'f8', ('z',))[:] = dz
    d.createVariable('topog', 'f8', ('y', 'x'))[:] = -h
    d.createVariable('taux',  'f8', ('y', 'x'))[:] = tau_x
    d.createVariable('sst',   'f8', ('y', 'x'))[:] = SST
    d.createVariable('salin', 'f8', ('y', 'x'))[:] = Fs
