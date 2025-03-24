import numpy as np
import toml
class PerformanceModel:
    '''Performance model for an aircraft. Contains
        fuel flow, airspeed, ROC/ROD, LTO emissions,
        and OAG schedule'''

    def __init__(self, config_file_loc="./IO/default_config.toml"):
        # Set headers for state variable NumPy structured array
        self.dtype = [
            ('t', float), # time (s)
            ('h', float), # Altitude (ft)
            ('gamma', float), # Flight Path angle: Cruise gamma =0, climb gamma >0, descent gamma < 0
            ('airspeed', float), # TAS 
            ('fuel_mass', float), # Mass of fuel
            ('fuel_rate', float) # Fuel flow rate
        ]
        # Read config file and store all variables in self.config
        self.config = {}
        with open(config_file_loc, 'r') as f:
            config_data = toml.load(f)
            self.config = {k: v for subdict in config_data.values() for k, v in subdict.items()}
        # Initialize state variables as a numpy structured array
        self.states = np.empty(0, dtype=self.dtype)