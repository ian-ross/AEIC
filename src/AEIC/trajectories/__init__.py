from collections.abc import Iterator

import numpy as np
import xarray as xr

TRAJECTORY_FIELDS = {
    'fuelFlow': {'description': 'Fuel flow rate', 'units': 'kg/s'},
    'acMass': {'description': 'Aircraft mass', 'units': 'kg'},
    'fuelMass': {'description': 'Fuel mass remaining', 'units': 'kg'},
    'groundDist': {'description': 'Ground distance traveled', 'units': 'm'},
    'altitude': {'description': 'Altitude above sea level', 'units': 'm'},
    'FLs': {'description': 'Flight level', 'units': 'FL'},
    'rocs': {'description': 'Rate of climb/descent', 'units': 'm/s'},
    'flightTime': {'description': 'Flight time elapsed', 'units': 's'},
    'latitude': {'description': 'Latitude', 'units': 'degrees'},
    'longitude': {'description': 'Longitude', 'units': 'degrees'},
    'azimuth': {'description': 'Azimuth angle', 'units': 'degrees'},
    'heading': {'description': 'Aircraft heading', 'units': 'degrees'},
    'tas': {'description': 'True airspeed', 'units': 'm/s'},
    'groundSpeed': {'description': 'Ground speed', 'units': 'm/s'},
    'FL_weight': {
        'description': 'Flight level weight factor',
        'units': 'dimensionless',
    },
}


class Trajectory:
    def __init__(self, npoints: int, name: str | None = None):
        self.npoints = npoints
        self.name = name
        self.data = {
            name: np.zeros((npoints,), dtype=float) for name in TRAJECTORY_FIELDS.keys()
        }

    def __len__(self):
        return self.npoints

    def __getattr__(self, name):
        if name in TRAJECTORY_FIELDS:
            return self.data[name]
        raise AttributeError(f"'Trajectory' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in TRAJECTORY_FIELDS:
            if len(value) != self.npoints:
                raise ValueError('Assigned length does not match number of points')
            self.data[name] = np.asarray(value, dtype=float)
        else:
            super().__setattr__(name, value)

    def copy_point(self, from_idx: int, to_idx: int):
        if from_idx < 0 or from_idx >= self.npoints:
            raise IndexError('from_idx out of range')
        if to_idx < 0 or to_idx >= self.npoints:
            raise IndexError('to_idx out of range')
        for name in TRAJECTORY_FIELDS.keys():
            self.data[name][to_idx] = self.data[name][from_idx]


class _TrajectorySetIterator(Iterator):
    def __init__(self, set):
        self._set = set
        self._index = 0

    def __next__(self):
        if self._index < len(self._set):
            item = self._set[self._index]
            self._index += 1
            return item
        else:
            raise StopIteration


class TrajectorySet:
    def __init__(self, name: str | None = None):
        self.name = name
        self._trajectories: list[Trajectory] = []

    def __len__(self):
        return len(self._trajectories)

    def __getitem__(self, idx) -> Trajectory:
        if idx < 0 or idx >= len(self):
            raise IndexError('Trajectory index out of range')
        return self._trajectories[idx]

    def __iter__(self) -> Iterator[Trajectory]:
        return _TrajectorySetIterator(self)

    def add(self, trajectory: Trajectory) -> int:
        self._trajectories.append(trajectory)
        return len(self._trajectories) - 1

    def to_xarray(self):
        # Calculate number of points and starting indices across trajectories.
        npointss = [len(t) for t in self._trajectories]
        starts = np.cumsum([0] + npointss[:-1])
        names = [t.name or '' for t in self._trajectories]

        # Dataset with coordinates and trajectory bookkeeping variables.
        ds = xr.Dataset(
            coords=dict(
                point=('point', np.arange(sum(npointss))),
                trajectory=('trajectory', np.arange(len(self._trajectories))),
            ),
            attrs=dict(
                title='Aircraft trajectory data',
                name=self.name or 'anonymous trajectory set',
                description='1D trajectories over mission points',
            ),
            data_vars=dict(
                npoints=(
                    'trajectory',
                    npointss,
                    {
                        'description': 'Number of points in each trajectory',
                        'units': 'count',
                    },
                ),
                start=(
                    'trajectory',
                    starts,
                    {
                        'description': (
                            'Starting index of each trajectory in the point dimension'
                        ),
                        'units': 'index',
                    },
                ),
                names=(
                    'trajectory',
                    names,
                    {'description': 'Names of each trajectory', 'units': 'string'},
                ),
            ),
        )

        # TODO: Add other application-specific global attributes here as
        # needed.

        # TODO: Add other global or variable-level CF attributes here as
        # needed.

        # Add data variables with metadata.
        for var_name, metadata in TRAJECTORY_FIELDS.items():
            ds[var_name] = (
                'point',
                np.concatenate([getattr(t, var_name) for t in self._trajectories]),
            )
            ds[var_name].attrs['description'] = metadata['description']
            ds[var_name].attrs['units'] = metadata['units']

        return ds

    @classmethod
    def from_xarray(cls, ds: xr.Dataset) -> 'TrajectorySet':
        result = cls()
        for itraj in range(ds.sizes['trajectory']):
            start = int(ds['start'].values[itraj])
            npoints = int(ds['npoints'].values[itraj])
            name = str(ds['names'].values[itraj])
            traj = Trajectory(npoints, name if name != '' else None)
            for var_name in TRAJECTORY_FIELDS.keys():
                traj.data[var_name] = ds[var_name].values[start : start + npoints]
            result.add(traj)

        # TODO: Rationalize handling of global attributes.
        if 'name' in ds.attrs:
            result.name = ds.attrs['name']

        return result

    def to_netcdf(self, path):
        self.to_xarray().to_netcdf(path)

    @classmethod
    def from_netcdf(cls, path) -> 'TrajectorySet':
        return cls.from_xarray(xr.load_dataset(path))
