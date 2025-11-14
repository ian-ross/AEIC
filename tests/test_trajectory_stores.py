from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from AEIC.trajectories.field_sets import FieldMetadata, FieldSet
from AEIC.trajectories.store import TrajectoryStore
from AEIC.trajectories.trajectory import Trajectory


@dataclass
class Extras:
    FIELD_SET: ClassVar[FieldSet] = FieldSet(
        'demo',
        f1=FieldMetadata(description='Test 1', units='unit1'),
        f2=FieldMetadata(description='Test 2', units='unit2'),
        mf=FieldMetadata(
            metadata=True,
            field_type=np.int32,
            description='Test metadata',
            units='unit3',
        ),
    )

    f1: np.ndarray
    f2: np.ndarray
    mf: int

    @classmethod
    def random(cls, npoints: int) -> 'Extras':
        return cls(
            f1=np.random.rand(npoints),
            f2=np.random.rand(npoints),
            mf=np.random.randint(0, 100),
        )


def make_test_trajectory(npoints: int, seed: int) -> Trajectory:
    t = Trajectory(npoints, name=f'traj_{seed}')
    t.fuelFlow = np.random.rand(npoints) * 5000 + 2000
    t.acMass = np.random.rand(npoints) * 50000 + 100000
    t.fuelMass = np.random.rand(npoints) * 50000 + 5000
    t.groundDist = np.linspace(0, 1000, npoints)
    t.altitude = np.linspace(0, 35000, npoints)
    t.FLs = t.altitude / 100
    t.rocs = np.random.randn(npoints) * 10
    t.flightTime = np.linspace(0, 3600, npoints)
    t.latitude = np.random.rand(npoints) * 180 - 90
    t.longitude = np.random.rand(npoints) * 360 - 180
    t.azimuth = np.random.rand(npoints) * 360
    t.heading = np.random.rand(npoints) * 360
    t.tas = np.random.rand(npoints) * 300 + 200
    t.groundSpeed = t.tas + np.random.randn(npoints) * 5
    t.FL_weight = np.ones(npoints)
    t.starting_mass = t.acMass[0]
    t.fuel_mass = t.fuelMass[0] - t.fuelMass[-1]
    t.NClm = npoints // 3
    t.NCrz = npoints // 3
    t.NDes = npoints - t.NClm - t.NCrz
    return t


def test_use_case_1():
    # USE CASE 1: create a small TrajectoryStore, save to NetCDF, disabling
    # further appending (closes NetCDF file), reload from NetCDF.

    Path('use_case_1.nc').unlink(missing_ok=True)
    tset = TrajectoryStore(
        title='Use case 1',
        nc_file='use_case_1.nc',
        mode=TrajectoryStore.FileMode.CREATE,
    )
    tset.add(make_test_trajectory(10, 1))
    tset.add(make_test_trajectory(15, 2))
    tset.close()

    tset_loaded = TrajectoryStore(nc_file='use_case_1.nc')
    assert tset_loaded.global_attributes['title'] == 'Use case 1'
    assert len(tset_loaded) == 2
    assert len(tset_loaded[0]) == 10
    assert len(tset_loaded[1]) == 15


def test_use_case_2():
    # USE CASE 2: create a TrajectoryStore, save to NetCDF, close, reopen file
    # in append mode, add another trajectory, close NetCDF file and reload from
    # NetCDF.

    tset2 = TrajectoryStore(
        title='Use case 2',
        nc_file='use_case_2.nc',
        mode=TrajectoryStore.FileMode.CREATE,
    )
    tset2.add(Trajectory(10, name='traj1'))
    tset2.add(Trajectory(15, name='traj2'))
    tset2.close()

    tset2a = TrajectoryStore(
        nc_file='use_case_2.nc', mode=TrajectoryStore.FileMode.APPEND
    )
    tset2a.add(Trajectory(20, name='traj3'))
    tset2a.close()

    tset2_loaded = TrajectoryStore(nc_file='use_case_2.nc')
    assert tset2_loaded.global_attributes['title'] == 'Use case 2'
    assert len(tset2_loaded) == 3
    assert len(tset2_loaded[0]) == 10
    assert len(tset2_loaded[1]) == 15
    assert len(tset2_loaded[2]) == 20


def test_use_case_3():
    # USE CASE 3: create large TrajectoryStore linked with NetCDF file for
    # writing, close the NetCDF file, reopen for reading and check contents.

    tset3 = TrajectoryStore(
        title='Use case 3',
        nc_file='use_case_3.nc',
        mode=TrajectoryStore.FileMode.CREATE,
    )
    for i in range(1000000):
        tset3.add(Trajectory(100, name=f'traj_{i}'))
    tset3.close()

    tset3_loaded = TrajectoryStore(nc_file='use_case_3.nc')
    assert len(tset3_loaded) == 1000000
    assert len(tset3_loaded[200000]) == 100
    assert len(tset3_loaded[999999]) == 100


def test_use_case_4():
    # USE CASE 4: create TrajectoryStore with additional field set saved in
    # base file. (This should result in a file with a "base" group and a "demo"
    # group.)

    tset4 = TrajectoryStore(
        title='Use case 4',
        nc_file='use_case_4.nc',
        mode=TrajectoryStore.FileMode.CREATE,
    )
    for i in range(1, 6):
        t = Trajectory(i * 5, name=f'traj{i}')
        t.add_fields(Extras.random(i * 5))
        tset4.add(t)
    tset4.close()

    tset4_loaded = TrajectoryStore(nc_file='use_case_4.nc')
    assert len(tset4_loaded) == 5
    assert tset4_loaded.fieldsets == {'base', 'demo'}
    assert tset4_loaded[2].f1.shape == (15,)
    assert tset4_loaded[2].mf is not None


# TODO: Related use case — add fields only to a sub-set of trajectories. Should
# result in an error from the data dictionary hash check in
# TrajectoryStore.add().


def test_use_case_5():
    # USE CASE 5: same thing, except save additional field set in an associated
    # file.

    tset5 = TrajectoryStore(
        title='Use case 5',
        nc_file='use_case_5.nc',
        mode=TrajectoryStore.FileMode.CREATE,
        associated_nc_files=[('demo_fields.nc', ['demo'])],
    )
    for i in range(1, 6):
        t = Trajectory(i * 5, name=f'traj{i}')
        t.add_fields(Extras.random(i * 5))
        tset5.add(t)
    tset5.close()

    # TODO: Append to the files here...

    tset5_loaded = TrajectoryStore(nc_file='use_case_5.nc')
    assert len(tset5_loaded) == 5
    assert tset5_loaded.fieldsets == {'base'}
    assert tset5_loaded[2].f1.shape == (15,)
    assert tset5_loaded[2].mf is not None

    tset5a_loaded = TrajectoryStore(
        nc_file='use_case_5.nc', associated_nc_files=['demo_fields.nc']
    )
    assert len(tset5a_loaded) == 5
    assert tset5a_loaded.fieldsets == {'base', 'demo'}
    assert tset5a_loaded[2].f1.shape == (15,)
    assert tset5a_loaded[2].mf is not None


def test_use_case_6():
    # USE CASE 6: create associated file from existing TrajectoryStore.

    # TODO: Is this essentially a map over trajectories? You need to fill in
    # the associated data for each trajectory, so you basically need a function
    # taking a trajectory and returning data conformable with a given field
    # set.
    #
    # That makes it sound like you don't need a CREATE_ASSOCIATED mode: just
    # open the input file in READ mode, then call a method on TrajectoryStore
    # that does the mapping and creates the associated file. Once you have an
    # associated file, you can just reopen the TrajectoryStore in READ or
    # APPEND mode.
    ...
