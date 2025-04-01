import numpy as np
from AEIC.performance_model import PerformanceModel

class Trajectory:
    '''Model for determining flight trajectories.
    '''
    
    def __init__(self, ac_performance:PerformanceModel):
        self.ac_performance = ac_performance
    
    
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