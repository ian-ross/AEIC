import AEIC.trajectories.builders as tb
from AEIC.performance_model import PerformanceModel
from AEIC.trajectories import TrajectorySet
from utils import file_location


def test_trajectory_simulation_1():
    performance_model_file = file_location("IO/default_config.toml")
    perf = PerformanceModel(performance_model_file)
    builder = tb.LegacyBuilder(options=tb.Options(iterate_mass=False))
    ts = TrajectorySet()

    for mis in perf.missions:
        traj = builder.fly(perf, mis)
        ts.add(traj)

    ts.to_netcdf("test_trajectories.nc")

    ts_loaded = TrajectorySet.from_netcdf("test_trajectories.nc")
    assert len(ts_loaded) == len(ts)
