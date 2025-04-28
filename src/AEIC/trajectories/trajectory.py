import numpy as np
from AEIC.performance_model import PerformanceModel
from src.utils.helpers import nautmiles_to_meters

class Trajectory:
    '''Model for determining flight trajectories.
    '''
    
    def __init__(self, ac_performance:PerformanceModel, mission, optimize:bool):
        # Save A/C performance model and the mission to be flown
        # NOTE: Currently assume that `mission` comes in as a dictionary with the format of a single flight
        # in `src/missions/sample_missions_10.json`. We also assume that Load Factor for the flight will be
        # included in the mission object.
        self.name = f'{mission["dep_airport"]}_{mission["arr_airport"]}_{mission["ac_code"]}'
        self.ac_performance = ac_performance
        
        # Save airport locations and dep/arr times; lat/long in degrees
        self.dep_lon_lat_alt = mission['dep_location']
        self.arr_lon_lat_alt = mission['arr_location']
        
        self.start_time = mission["dep_datetime"]
        self.end_time = mission["arr_datetime"]
        
        # Convert gc distance to meters
        self.gc_distance = mission["distance_nm"] # FIXME: check to make sure this is changed to meters
    
        # Get load factor from mission object
        self.load_factor = mission["load_factor"]
    
        # Controls whether or not route optimization is performed
        # NOTE: This currently does nothing
        self.optimize = optimize        
    
    
    def climb(self):
        pass
    
    
    def cruise(self):
        pass
    
    
    def descent(self):
        pass
    
    
    def lto(self):
        pass