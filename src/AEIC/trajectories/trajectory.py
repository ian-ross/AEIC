import numpy as np
from AEIC.performance_model import PerformanceModel

class Trajectory:
    '''Model for determining flight trajectories.
    '''
    
    def __init__(self, ac_performance:PerformanceModel, optimize:bool):
        self.ac_performance = ac_performance
        # Need two types of performance tables:
        #
        # 1. A table encoding nominal mission: (fuel flow, ROC, TAS) = f(alt, a/c mass)
        #
        # 2. A 4D numpy array (m x n x k x l) where the dimensions are the number of alt, ROC, TAS, and a/c mass, respectively
        #    -> This should also include the individual value sets (alt = np.array(m), ROC = np.array(n), TAS = np.array(k), mass = np.array(l))
        #
        #
        # (1) can be used for getting a nominal mission and is final result for BADA-3 input. (2) Is used by the weight/vertical iteration loop
        # to determine an optimal vertical trajectory and approximation of fuel mass. 
    
    
    def calc_trajectory(self):
        '''
        Uses the `climb`, `cruise`, `descent`, and `lto` functions
        defined by child classes (specific trajectory calculation
        methods) to determine a flight's trajectory.
        
        Outputs:
        - NDArray containing (time, altitude, lat, lon, fuel flow)
        '''
        # Optimize horizontal if wanted (estimate TOC alt from ADS-B sampling or nominal mission)
        #
        # Intermediate for eletra: atmospheric data along horizontal trajectory
        #
        # Loop to deterimine fuel mass/vertical optimization
        #   -> Optimize a mission with max fuel mass
        #   -> Subtract excess fuel left at end from intial fuel
        #   -> Repeat a couple times until remaining fuel is close to 0 (~10kg, 
        #      ~100kg, doesn't have to be perfect)
        #
        # Cruise vertical optimization to be done using Marek's code
        #
        # Climb/Descent need constraint specification (max ROC, maximum climb time, etc.)
        #
        pass
    
    
    def climb(self):
        pass
    
    
    def cruise(self):
        pass
    
    
    def descent(self):
        pass
    
    
    def lto(self):
        pass