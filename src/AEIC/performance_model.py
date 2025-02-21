import numpy as np
import toml 
class PerformanceModel:
    '''Performance model for an aircraft. Contains
        fuel flow, airspeed, ROC/ROD, LTO emissions,
        and OAG schedule'''

    def __init__(self, config_file_loc=None):
        self.dtype = [
            ('h', float),
            ('airspeed', float),
            ('gamma', float),
            ('fuel_mass', float),
            ('fuel_rate', float)
        ]
        with open(config_file_loc, 'r') as f:
            self.config = toml.load(f)
        self.states = np.empty(0, dtype=self.dtype)
        self.initialize()

    def initialize(self):
        '''Uses config file to setup performance
           model data'''
        
        if self.config is None:
            print("No config file provided - will not intialize")
            pass
        else:
            print("Reading configuration file ...")
            self.read_config()

            print("Creating Schedule ... ")
            self.create_schedule()

            print("Reading LTO data ... ")
            self.read_LTO()

            print("Reading Non LTO data ... ")
            self.read_non_LTO()

    def read_LTO(self):
        '''Reads and stores LTO data'''
        pass

    def read_non_LTO(self):
        '''Reads and stores non LTO data'''
        pass

    def create_schedule(self):
        '''Reads and stores OAG Schedule data'''
        pass

    def read_config(self):
       """
        Reads through each section/key in self.config
        and sets them as attributes on the class instance.
        """
       for _, params in self.config.items():
            # Each 'params' is another dict of key-value pairs
            for key, value in params.items():
                setattr(self, key, value)
