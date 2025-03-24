import numpy as np
import toml
import json
from src.parsers.PTF_reader import parse_PTF
class PerformanceModel:
    '''Performance model for an aircraft. Contains
        fuel flow, airspeed, ROC/ROD, LTO emissions,
        and OAG schedule'''

    def __init__(self, config_file_loc="./IO/default_config.toml"):
        # Set headers for state variable NumPy structured array
        self.dtype = [
            ('h', float),            # Altitude (ft)
            ('rocd', float, (2,)),   # Rate of climb/descent (2 elements: low & high)
            ('airspeed', float),     # TAS
            ('fuel_rate', float, (2,)) # Fuel flow rate (2 elements: low & high)
        ]
        # Read config file and store all variables in self.config
        self.config = {}
        with open(config_file_loc, 'r') as f:
            config_data = toml.load(f)
            self.config = {k: v for subdict in config_data.values() for k, v in subdict.items()}
        # Initialize state variables as a numpy structured array
        self.states = np.empty(0, dtype=self.dtype)

        # Process input performance data
        self.initialize_performance()

    def initialize_performance(self):
        '''Takes input data given on aircraft performance
            and creates the state variable array'''
        
        # If OPF data input
        if self.config["performance_model_input"] == "OPF":
            #opf_data = parse_OPF(self.config["performance_model_input"])
            pass
        # If PTF data input
        elif self.config["performance_model_input"] == "PTF":
            ptf_data = parse_PTF(self.config["performance_model_input_file"])
            # Now build self.states from ptf_data["phases"] exactly like read_performance_data does
            phases = ptf_data.pop("phases", {})
            
            phase_arrays = []
            for phase_name, phase_data in phases.items():
                flight_levels = np.array(phase_data['flight_levels_ft'], dtype=float)
                rocd_lo       = np.array(phase_data['rocd_lo'],         dtype=float)
                rocd_hi       = np.array(phase_data['rocd_hi'],         dtype=float)
                tas           = np.array(phase_data['tas'],             dtype=float)
                fuel_flow_lo  = np.array(phase_data['fuel_flow_lo'],    dtype=float)
                fuel_flow_hi  = np.array(phase_data['fuel_flow_hi'],    dtype=float)
                
                rocd       = np.column_stack((rocd_lo, rocd_hi))
                fuel_flow  = np.column_stack((fuel_flow_lo, fuel_flow_hi))

                # Create a structured array for this phase
                phase_array = np.zeros(len(flight_levels), dtype=self.dtype)
                phase_array['h']         = flight_levels
                phase_array['rocd']      = rocd
                phase_array['airspeed']  = tas
                phase_array['fuel_rate'] = fuel_flow

                phase_arrays.append(phase_array)
            
            # Concatenate them
            if phase_arrays:
                self.states = np.concatenate(phase_arrays)
            
            # Store the rest of the info in self.model_info
            self.model_info = {}
            self.model_info.update(ptf_data)
        # If fuel flow function input
        else:
            self.read_performance_data()
            

    def read_performance_data(self):
        '''Parses input json data of aircraft performance'''
        
        # Read and load JSON data 
        with open(self.config["performance_model_input_file"], 'r') as f:
            data = json.load(f)

        # Extract the 'phases' dictionary and remove it from 'data'
        phases = data.pop("phases", {})

        # Prepare a list to collect structured arrays from each phase.
        phase_arrays = []

        # Loop through each phase and build a structured array in bulk.
        for _, phase_data in phases.items():
            flight_levels = np.array(phase_data['flight_levels_ft'], dtype=float)
            rocd_lo       = np.array(phase_data['rocd_lo'],          dtype=float)
            rocd_hi       = np.array(phase_data['rocd_hi'],          dtype=float)
            tas           = np.array(phase_data['tas'],              dtype=float)
            fuel_flow_lo  = np.array(phase_data['fuel_flow_lo'],     dtype=float)
            fuel_flow_hi  = np.array(phase_data['fuel_flow_hi'],     dtype=float)

            # Combine 'rocd_lo' and 'rocd_hi' columns into a single Nx2 array.
            rocd = np.column_stack((rocd_lo, rocd_hi))

            # Combine 'fuel_flow_lo' and 'fuel_flow_hi' columns into a single Nx2 array.
            fuel_flow = np.column_stack((fuel_flow_lo, fuel_flow_hi))

            # Create a new structured array for all rows of this phase at once.
            # This avoids the overhead of appending row-by-row.
            phase_array = np.zeros(len(flight_levels), dtype=self.dtype)
            phase_array['h']         = flight_levels
            phase_array['rocd']      = rocd
            phase_array['airspeed']  = tas
            phase_array['fuel_rate'] = fuel_flow

            # Collect this phase's array.
            phase_arrays.append(phase_array)

        # Concatenate all per-phase arrays into one final structured array.
        self.states = np.concatenate(phase_arrays)
        # Store rest of data provided
        self.model_info = {}
        self.model_info.update(data)
