import numpy as np

import AEIC.trajectories.builders as tb
from AEIC.performance_model import PerformanceModel
from AEIC.trajectories import FieldMetadata, FieldSet, TrajectoryStore
from utils import file_location

test_fields = FieldSet(
    "test_fields",
    test_field1=FieldMetadata(
        field_type=float, description="A test field 1", units="unit1"
    ),
    test_field2=FieldMetadata(
        field_type=int, description="A test field 2", units="unit2"
    ),
)


def test_trajectory_simulation_1():
    performance_model_file = file_location("IO/default_config.toml")
    perf = PerformanceModel(performance_model_file)
    builder = tb.LegacyBuilder(options=tb.Options(iterate_mass=False))
    ts = TrajectoryStore(
        nc_file='test_trajectories.nc', mode=TrajectoryStore.FileMode.CREATE
    )

    for mis in perf.missions:
        traj = builder.fly(perf, mis)
        traj.add_fields(test_fields)
        traj.test_field1 = np.random.rand(len(traj))
        traj.test_field2 = np.random.randint(0, 100, size=len(traj))
        ts.add(traj)

    ts.close()

    ts_loaded = TrajectoryStore(nc_file="test_trajectories.nc")
    assert len(ts_loaded) == len(ts)
    # TODO: Test that additional fields are correctly saved and loaded.
