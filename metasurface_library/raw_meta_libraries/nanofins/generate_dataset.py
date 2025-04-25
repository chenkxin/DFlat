import numpy as np
import scipy.io as sio
import mat73
import math

# Define parameters
lenx = np.linspace(60, 300, 49).reshape(49, 1) *1e-9
leny = np.linspace(60, 300, 49).reshape(49, 1) *1e-9

phasex=mat73.loadmat('phasex.mat')['phase']
phasey=mat73.loadmat('phasey.mat')['phase']
phasex=np.expand_dims(phasex, axis=0)
phasey =np.expand_dims(phasey, axis=0)


Tx=mat73.loadmat('Tx.mat')['T']
Ty=mat73.loadmat('Ty.mat')['T']
Tx=np.expand_dims(Tx, axis=0)
Ty =np.expand_dims(Ty, axis=0)

phase = np.vstack((phasex, phasey))
transmission = np.vstack((Tx, Ty))


# Create meshgrid
wavelength_m = np.linspace(310, 750, 441).reshape(441, 1) *1e-9


sio.savemat('data_Nanofins_Unit350nm_Height600nm_EngineFDTD_test.mat', {'lenx': lenx,'leny': leny, 'phase': phase, 'transmission': transmission, 'wavelength_m': wavelength_m})

