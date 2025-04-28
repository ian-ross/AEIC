import numpy as np
from AEIC.performance_model import PerformanceModel
from AEIC.trajectories.trajectory import Trajectory
from utils.helpers import feet_to_meters

class LegacyTrajectory(Trajectory):
    '''Model for determining flight trajectories using the legacy method
    from AEIC v2.
    '''
    def __init__(self, ac_performance:PerformanceModel):
        super().__init__(ac_performance)
        
        # Climb defined as starting 3000' above airport
        self.current_altitude = self.dep_lon_lat_alt[-1] + feet_to_meters(3000.0)
        
        
    
    
    
    
    def climb(self):
        pass
    
    
    def cruise(self):
        pass
    
    
    def descent(self):
        pass

    
    