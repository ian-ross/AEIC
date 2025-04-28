import numpy as np
from AEIC.performance_model import PerformanceModel
from src.utils.helpers import nautmiles_to_meters

class Trajectory:
    '''Model for determining flight trajectories.
    '''
    
    def __init__(self, ac_performance:PerformanceModel, mission, optimize:bool):
        # Save A/C performance model and the mission to be flown
        # NOTE: Currently assume that `mission` comes in as a dictionary with the format of a single flight
        # in `src/missions/sample_missions_10.json`
        self.name = f'{mission["dep_airport"]}_{mission["arr_airport"]}_{mission["ac_code"]}'
        self.ac_performance = ac_performance
        
        # Save airport locations and dep/arr times; lat/long in degrees
        self.dep_lon_lat_alt = mission['dep_location']
        self.arr_lon_lat_alt = mission['arr_location']
        
        self.start_time = mission["dep_datetime"]
        self.end_time = mission["arr_datetime"]
        
        # Convert gc distance to meters
        self.gc_distance = nautmiles_to_meters(mission["distance_nm"]) 
    
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