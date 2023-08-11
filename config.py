# Kenny Schlegel, Peer Neubert, Peter Protzel
#
# HDC-MiniROCKET : Explicit Time Encoding in Time Series Classification with Hyperdimensional Computing
# Copyright (C) 2022 Chair of Automation Technology / TU Chemnitz

import numpy as np

def default_init(self):
    self.norm_hdc_output = True
    self.best_scale = False
    self.scales = np.logspace(0,3,7,base=2)-1

    # general parameters
    self.n_time_measures = 1
    self.seed = 0

    return self

class Config_orig(object):
    """
    configuration for classification on defined scale
    """
    def __init__(self):
        # HDC Minirocket Config
        self = default_init(self)
        self.note = ''

class Config_orig_auto(object):
    """
    configuration for classification with automatically selected the best scaling parameter (grid search)
    """
    def __init__(self):
        # HDC Minirocket Config
        self = default_init(self)
        self.best_scale = True
        self.note = 'auto'

class Config_time_measure(object):
    """
    configuration for classification with time measuring
    """
    def __init__(self):
        # HDC Minirocket Config
        self = default_init(self)
        self.n_time_measures = 5
        self.note = 'time_measure'