#io_utils.py
import numpy as np
import h5py


def _to_numpy(array):
	return array.get() if hasattr(array, 'get') else np.asarray(array)


def save_hdf5(parameters, vel_u, vel_v, pressure):
	with h5py.File('data.hdf5', 'w') as f:
		#Parameters
		for key, val in parameters.items():
			f.attrs[key] = val

		#Fields
		dset = f.create_dataset('vel_u', data=_to_numpy(vel_u))
		dset = f.create_dataset('vel_v', data=_to_numpy(vel_v))
		dset = f.create_dataset('pressure', data=_to_numpy(pressure))


def load_hdf5():
	with h5py.File('data.hdf5', 'r') as f:
		#Parameters
		parameters = dict(f.attrs)

		#Fields
		vel_u = f['vel_u'][:]
		vel_v = f['vel_v'][:]
		pressure = f['pressure'][:]

	return parameters, vel_u, vel_v, pressure