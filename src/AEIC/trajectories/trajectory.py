import numpy as np
from AEIC.performance_model import PerformanceModel

class Trajectory:
    '''Model for determining flight trajectories.
    '''
    
    def __init__(self, ac_performance:PerformanceModel, mission, optimize:bool):
        # Save A/C performance model and the mission to be flown
        # NOTE: Currently assume that `mission` comes in as a dictionary with the format of a single flight
        # in `src/missions/sample_missions_10.json`
        self.name = f'{mission["dep_airport"]}_{mission["arr_airport"]}_{mission["ac_code"]}'
        self.ac_performance = ac_performance
        self.mission = mission
        
        # Controls whether or not route optimization is performed
        # NOTE: This currently does nothing
        self.optimize = optimize
        
    
    
    def calc_trajectory(self):
        '''
        Uses the `climb`, `cruise`, `descent`, and `lto` functions
        defined by child classes (specific trajectory calculation
        methods) to determine a flight's trajectory.
        
        Outputs:
        - NDArray containing (time, altitude, lat, lon, fuel flow)
        '''
        pass
    
    
    def climb(self):
        pass
    
    
    def cruise(self):
        pass
    
    
    def descent(self):
        pass
    
    
    def lto(self):
        pass