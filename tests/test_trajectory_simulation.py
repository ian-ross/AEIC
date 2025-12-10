import tomllib

import numpy as np
import pytest

import AEIC.trajectories.builders as tb
from AEIC.missions import Mission
from AEIC.performance_model import PerformanceModel
from AEIC.trajectories import FieldMetadata, FieldSet, TrajectoryStore
from AEIC.utils.files import file_location

test_fields = FieldSet(
    'test_fields',
    test_field1=FieldMetadata(
        field_type=np.float32, description='A test field 1', units='unit1'
    ),
    test_field2=FieldMetadata(
        field_type=np.int32, description='A test field 2', units='unit2'
    ),
)


def test_trajectory_simulation_1(tmp_path):
    fname = tmp_path / 'test_trajectories.nc'
    performance_model_file = file_location('IO/default_config.toml')
    missions_file = file_location('missions/sample_missions_10.toml')

    perf = PerformanceModel(performance_model_file)
    builder = tb.LegacyBuilder(options=tb.Options(iterate_mass=False))
    ts = TrajectoryStore.create(base_file=fname)
    # Load mission toml into dict
    mission_file = file_location(missions_file)
    with open(mission_file, 'rb') as f:
        mission_dict = tomllib.load(f)
    missions = Mission.from_toml(mission_dict)

    for mis in missions:
        traj = builder.fly(perf, mis)
        traj.add_fields(test_fields)
        traj.test_field1 = np.random.rand(len(traj))
        traj.test_field2 = np.random.randint(0, 100, size=len(traj))
        ts.add(traj)

    ts.close()

    ts_loaded = TrajectoryStore.open(base_file=fname)
    assert len(ts_loaded) == len(ts)
    # TODO: Test that additional fields are correctly saved and loaded.


def test_trajectory_mass_iter():
    """Test that:
    - Mass iteration is consistent between start and end of mission
    - The final mass residual is less than required for convergence
    - In case without enough iterations to meet reltol, error is raised
    """
    test_reltol = 1e-6
    test_maxiters = 1000

    test_mis = Mission(
        origin='BOS',
        destination='LAX',
        departure="2019-01-01T12:00:00",
        arrival="2019-01-01T18:00:00",
        aircraft_type="738",
        load_factor=1.0,
    )

    perf = PerformanceModel(file_location('IO/default_config.toml'))

    builder_wout_iter = tb.LegacyBuilder(options=tb.Options(iterate_mass=False))

    builder_with_iter = tb.LegacyBuilder(
        options=tb.Options(
            iterate_mass=True,
            max_mass_iters=test_maxiters,
            mass_iter_reltol=test_reltol,
        )
    )

    builder_fail = tb.LegacyBuilder(
        options=tb.Options(
            iterate_mass=True, max_mass_iters=1, mass_iter_reltol=test_reltol
        )
    )

    traj_wout_iter = builder_wout_iter.fly(perf, test_mis)
    traj_with_iter = builder_with_iter.fly(perf, test_mis)

    fuel_difference = traj_wout_iter.fuel_mass[0] - traj_with_iter.fuel_mass[0]
    mass_difference = traj_wout_iter.aircraft_mass[0] - traj_with_iter.aircraft_mass[0]
    final_mass_residual = traj_with_iter.fuel_mass[-1] / traj_with_iter.fuel_mass[0]

    assert mass_difference == pytest.approx(fuel_difference)
    assert final_mass_residual < test_reltol

    with pytest.raises(RuntimeError):
        builder_fail.fly(perf, test_mis)
