import numpy as np
from AEIC.performance_model import PerformanceModel
from AEIC.trajectories.trajectory import Trajectory
from utils.helpers import feet_to_meters, meters_to_feet

class LegacyTrajectory(Trajectory):
    '''Model for determining flight trajectories using the legacy method
    from AEIC v2.
    '''
    def __init__(self, ac_performance:PerformanceModel, mission, optimize=False, startingMass=None):
        super().__init__(ac_performance)
        
        # Climb defined as starting 3000' above airport
        self.clm_start_altitude = self.dep_lon_lat_alt[-1] + feet_to_meters(3000.0)
        
        # Max alt should be changed to meters
        max_alt = feet_to_meters(ac_performance.model_info['General_Information']['max_alt_ft'])
        
        # Check if starting altitude is above operating ceiling; if true, set start altitude to
        # departure airport altitude
        if self.clm_start_altitude >= max_alt:
            self.clm_start_altitude = self.dep_lon_lat_alt[-1]
            
        # If flying the C172, take smaller climb steps
        if ac_performance['General_Information']['C172']:
            self.dAlt_climb = feet_to_meters(50.)
        else:
            self.dAlt_climb = feet_to_meters(1000.)
            
        # Cruise altitude is the operating ceiling - 7000 feet
        self.crz_altitude = max_alt - feet_to_meters(7000.)
        
        # Ensure cruise altitude is above the starting altitude
        if self.crz_altitude < self.clm_start_altitude:
            self.crz_altitude = self.clm_start_altitude
            
        # Prevent flying above A/C ceiling (NOTE: this will only trigger due to random 
        # variables not currently implemented)
        if self.crz_altitude > max_alt:
            self.crz_altitude = max_alt
        
        # Give the user the option to prescribe the aircraft starting mass
        self.start_mass = startingMass
        
        self.calc_mass = False    
        if self.start_mass is None:
            self.calc_mass = True
            
        # The legacy method precomputes starting mass (no iterations)
        self.iterate_mass = False
        
        # Save relevant flight levels
        self.crz_FL = meters_to_feet(self.crz_altitude) / 100
        self.clm_FL = meters_to_feet(self.clm_start_altitude) / 100
        
    
    def climb(self):
        pass
    
    
    def cruise(self):
        pass
    
    
    def descent(self):
        pass

    
    