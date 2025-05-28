import numpy as np
from src.AEIC.performance_model import PerformanceModel
from src.AEIC.trajectories.trajectory import Trajectory
from src.utils.helpers import feet_to_meters, meters_to_feet, nautmiles_to_meters

class LegacyTrajectory(Trajectory):
    '''Model for determining flight trajectories using the legacy method
    from AEIC v2.
    '''
    def __init__(self, ac_performance:PerformanceModel, mission, optimize_traj:bool, iterate_mass:bool, 
                 startMass:float=-1, pctStepClm=0.01, pctStepCrz=0.01, pctStepDes=0.01):
        super().__init__(ac_performance, mission, optimize_traj, iterate_mass, startMass=startMass)
        
        # Define discretization of each phase in steps as a percent of the overall distance/altitude change
        self.pctStepClm = pctStepClm
        self.pctStepCrz = pctStepCrz
        self.pctStepDes = pctStepDes
        
        self.NClm = 1 / self.pctStepClm + 1
        self.NCrz = 1 / self.pctStepCrz + 1
        self.NDes = 1 / self.pctStepDes + 1
        self.Ntot = self.NClm + self.NCrz + self.NDes
        
        # Climb defined as starting 3000' above airport
        self.clm_start_altitude = self.dep_lon_lat_alt[-1] + feet_to_meters(3000.0)
        
        # Max alt should be changed to meters
        max_alt = feet_to_meters(ac_performance.model_info['General_Information']['max_alt_ft'])
        
        # Check if starting altitude is above operating ceiling; if true, set start altitude to
        # departure airport altitude
        if self.clm_start_altitude >= max_alt:
            self.clm_start_altitude = self.dep_lon_lat_alt[-1]
            
        # Cruise altitude is the operating ceiling - 7000 feet
        self.crz_start_altitude = max_alt - feet_to_meters(7000.)
        
        # Ensure cruise altitude is above the starting altitude
        if self.crz_start_altitude < self.clm_start_altitude:
            self.crz_start_altitude = self.clm_start_altitude
            
        # Prevent flying above A/C ceiling (NOTE: this will only trigger due to random 
        # variables not currently implemented)
        if self.crz_start_altitude > max_alt:
            self.crz_start_altitude = max_alt
            
        # In legacy trajectory, descent start altitude is equal to cruise altitude
        self.des_start_altitude = self.crz_start_altitude
        
        # Set descent altitude based on 3000' above arrival airport altitude; clamp to A/C operating
        # ceiling if needed
        self.des_end_altitude = self.arr_lon_lat_alt[-1] + feet_to_meters(3000.0)
        if self.des_end_altitude >= max_alt:
            self.des_end_altitude = max_alt
        
        # Save relevant flight levels
        self.crz_FL = meters_to_feet(self.crz_start_altitude) / 100
        self.clm_FL = meters_to_feet(self.clm_start_altitude) / 100
        
        # Get the relevant bounding flight levels for cruise based on performance data
        self.__calc_crz_FLs()
        
        # Get the indices for 0-ROC performance
        self.__get_zero_roc_index()      
        
    
    def climb(self):
        pass
    
    
    def cruise(self):
        pass
    
    
    def descent(self):
        pass
    
    
    def lto(self):
        pass
    
    
    def calc_starting_mass(self):
        ''' Calculates the starting mass using AEIC v2 methods. Sets both starting mass and non-reserve/hold/divert fuel mass '''
        # Use the highest value of mass per AEIC v2 method
        mass_ind = [len(self.ac_performance.performance_table_cols[-1])-1]
        crz_mass = np.array(self.ac_performance.performance_table_cols[-1])[mass_ind]
        
        subset_performance = self.ac_performance.performance_table[
            np.ix_(self.crz_FL_inds,                                           # axis 0: flight levels
                   np.arange(self.ac_performance.performance_table.shape[1]),  # axis 1: all TAS's
                   np.where(self.roc_mask)[0],                                 # axis 2: ROC ≈ 0
                   mass_ind,                                                   # axis 3: high mass value
          )
        ]
    
        non_zero_mask = np.any(subset_performance != 0, axis=(0, 2, 3))
        non_zero_perf = subset_performance[:, non_zero_mask, :, :]
        crz_tas = np.array(self.ac_performance.performance_table_cols[1])[non_zero_mask]
        
        # At this point, we should have a (2, 2, 1, 1)-shape matrix of fuel flow in (FL, TAS, --, --)
        # where there should only be two non-0 values in the FL and TAS dimensions. Isolate this matrix:
        if np.shape(non_zero_perf) != (2,2,1,1):
            raise ValueError('Performance is overdefined for legacy methods')
        
        twoByTwoPerf = non_zero_perf[:,:,0,0]
        ff_mat = twoByTwoPerf[twoByTwoPerf != 0.0]
        if np.shape(ff_mat) != (2,):
            raise ValueError(f'Mass estimation fuel flow matrix does not have the required dimensions (Expected: (2,); Recieved: {np.shape(ff_mat)})')
        
        # Now perform the necessary interpolations in TAS and fuel flow
        FL_weighting = (self.crz_FL - self.crz_FLs[0]) / (self.crz_FLs[1] - self.crz_FLs[0])
        dfuelflow = ff_mat[1] - ff_mat[0]
        dTAS = crz_tas[1] - crz_tas[0]
        
        fuelflow = ff_mat[0] + dfuelflow * FL_weighting
        tas = crz_tas[0] + dTAS * FL_weighting
        
        # Figure out startingMass components per AEIC v2:
        #
        #      |   empty weight 
        #      | + payload weight
        #      | + fuel weight
        #      | + fuel reserves weight
        #      | + fuel divert weight
        #      | + fuel hold weight
        #      | _______________________
        #        = Take-off weight
        
        # Empty mass per BADA-3 (low mass / 1.2)
        emptyMass = self.ac_performance.performance_table_cols[-1][0] / 1.2
        
        # Payload
        payloadMass = self.ac_performance.model_info['General Information']['max_payload_kg'] * self.load_factor
        
        # Fuel Needed (distance / velocity * fuel flow rate)
        approxTime = self.gc_distance / tas
        fuelMass = approxTime * fuelflow
        
        # Reserve fuel (assumed 5%)
        reserveMass = fuelMass * 0.05
        
        # Diversion fuel per AEIC v2
        if approxTime / 60 > 180: # > 180 minutes
            divertMass = nautmiles_to_meters(200.) / tas * fuelflow
            holdMass = 30 * 60 * tas # 30 min; using cruise ff here
        else:
            divertMass = nautmiles_to_meters(100.) / tas * fuelflow
            holdMass = 45 * 60 * tas # 30 min; using cruise ff here
        
        self.starting_mass = emptyMass + payloadMass + fuelMass + reserveMass + divertMass + holdMass
        
        # Limit to MTOM if overweight
        if self.starting_mass > self.ac_performance.performance_table_cols[-1][-1]:
            self.starting_mass = self.ac_performance.performance_table_cols[-1][-1]        
        
        # Set fuel mass (for weight residual calculation)
        self.fuel_mass = fuelMass
        
        
    ###################
    # PRIVATE METHODS #
    ###################
    def __calc_crz_FLs(self):
        ''' Get the bounding cruise flight levels (for which data exists) '''
        # Get the two flight levels in data closest to the cruise FL
        FL_ind_high = np.searchsorted(self.ac_performance.performance_table_cols[0], self.crz_FL)
        
        if FL_ind_high == 0:
            raise ValueError('Aircraft is trying to fly below minimum cruise altitude')
        if FL_ind_high == len(self.ac_performance.performance_table_cols[0]) - 1:
            raise ValueError('Aircraft is trying to fly above maximum cruise altitude')
        
        FL_ind_low = FL_ind_high - 1
        self.crz_FL_inds = np.array([FL_ind_low, FL_ind_high])
        self.crz_FLs = np.array(self.ac_performance.performance_table_cols[0])[self.crz_FL_inds]
        
        
    def __get_zero_roc_index(self, roc_zero_tol=1e-6):
        ''' Get the index along the ROC axis of performance where ROC == 0 '''
        self.roc_mask = np.abs(np.array(self.ac_performance.performance_table_cols[2])) < roc_zero_tol