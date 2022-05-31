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

class Multiply():
    """
    Class definition for a multiplicative shift.
    """
    def __init__(self, bounds=(1E-2,1), scale="log", prior="uniform"):
        """
        Initialize the shift.
        Inputs:
            scale - either "log" for a shift in the logarithm of a variable or "linear" for the shift in the variable
        """
        self.scale = scale
        self.shared = False
        self.bounds = bounds
        self.default = 1
        self.type = "Multiply"
        if prior == "uniform":
            self.prior = lambda p, lam: 0
        elif prior == "Gaussian":
            if self.scale == "linear":
                self.prior = lambda p, lam: (lam**2)*(p**2)
            elif self.scale == "log":
                self.prior = lambda p, lam: (lam**2)*(np.log(p))**2

    def forward(self, param, state, data):
        """
        Run a forward shift on the data (and optinally the standard deviation.
        Inputs:
            param - value of the parameter in the shift (either "a" or "b")
            state - value of the state parameter for this data set
            data - coordinates (either x or y)
        Outputs:
            transformed - the transformed data
        """
        if self.scale == "log":
            transformed = data + np.log(param)
        else:
            transformed = data*param
        return transformed

    def backward(self, param, state, data):
        """
        Run a backward shift on the data (and optinally the standard deviation.
        Inputs:
            param - value of the parameter in the shift (either "a" or "b")
            state - value of the state parameter for this data set
            data - coordinates (either x or y)
        Outputs:
            transformed - the transformed data
        """
        if self.scale == "log":
            transformed = data - np.log(param)
        else:
            transformed = data/param
        return transformed
