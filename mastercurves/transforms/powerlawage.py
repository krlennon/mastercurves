# Copyright (c) 2022 Kyle R. Lennon (kyle.lennon08@gmail.com).
#
# This file is part of mastercurves.
#
# mastercurves is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
#
# mastercurves is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with mastercurves.
# If not, see <https://www.gnu.org/licenses/>.
#
# Please cite "A Data-Driven Method for Automated Data Superposition with Applications
# in Soft Matter Science" (https://arxiv.org/abs/2204.09521) if you use any part of the code.

import numpy as np

class PowerLawAge:
    """
    Class definition for a power law aging shift.
    """
    def __init__(self, tref, scale="log"):
        """
        Initialize the shift.
        Inputs:
            tref - the reference time
            scale - the scale of the time axis (log or linear)
        """
        self.param = "mu"
        self.scale = scale
        self.tref = tref
        self.shared = True
        self.bounds = (0.1,1.9)
        self.default = 1.1
        self.type = "PowerLawAge"
        self.prior = lambda p, lam: 0

    def forward(self, param, state, data):
        """
        Forward shift the data (from real time to effective time).
        Inputs:
            param - value of the shifting parameter mu
            state - value of the state parameter twait
            data - the time coordinate t (either t or log(t))
        Outputs:
            transformed - the effective time coordinate (either xi or log(xi))
        """
        if self.scale == "log":
            t = np.exp(data)
        else:
            t = data
        xi = (self.tref**param)*((t + state)**(1 - param) - state**(1 - param))/(1 - param)
        if self.scale == "log":
            transformed = np.log(xi)
        else:
            transformed = xi
        return transformed

    def backward(self, param, state, data):
        """
        Backward shift the data (from effective time to real time).
        Inputs:
            param - value of the shifting parameter mu
            state - value of the state parameter twait
            data - the effective time coordinate (either xi or log(xi))
        Outputs:
            transformed - the real time coordinate (either t or log(t))
        """
        if self.scale == "log":
            xi = np.exp(data)
        else:
            xi = data
        t = (xi*(1 - param)/(self.tref**param) + state**(1 - param))**(1/(1 - param)) - state
        if self.scale == "log":
            transformed = np.log(t)
        else:
            transformed = t
        return transformed
