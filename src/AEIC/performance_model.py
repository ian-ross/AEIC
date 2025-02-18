class PerformanceModel:
    '''Performance model for an aircraft. Contains
        fuel flow, airspeed, ROC/ROD, LTO emissions,
        and OAG schedule'''

    def __init__(self, config=None):
        self.config = config
        self.initialize()
        pass

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
        '''Reads and stores config data'''
        pass