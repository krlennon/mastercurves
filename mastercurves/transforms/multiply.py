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
    r"""
    Class definition for a multiplicative shift.

    Attributes:
        :attr:`bounds` (:attr:`tuple[float]`): the bounds for the shift factor

        :attr:`default` (:attr:`float`): default value of the shift factor (1)

        :attr:`prior` (:attr:`p, lam -> float`): the prior distribution over the shift
        factor. Either Gaussian or uniform.

        :attr:`scale` (:attr:`string`): coordinate scale, either log or linear

        :attr:`shared` (:attr:`bool`): :attr:`False`, since the shift factors
        are not shared between states

        :attr:`type` (:attr:`string`): :attr:`Multiply`
    """
    def __init__(self, bounds=(1E-2,1), scale="log", prior="uniform"):
        r"""
        Initialize the Multiply object.

        Args:
            :attr:`bounds` (:attr:`tuple[float]`): the upper and lower bounds for the shift
            factors. Defaults to (1E-2,1).

            :attr:`scale` (:attr:`string`): either "log" for a shift in the logarithm
            of a variable or "linear" for the shift in the variable. Defaults to "log".

            :attr:`prior` (:attr:`string`): either "uniform" for a uniform prior over the
            shift factors or "Gaussian" for a Gaussian prior. Defaults to "uniform".
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
        r"""
        Run a forward shift on the data (from the current state to the reference state).

        Args:
            :attr:`param` (:attr:`float`): value of the shift factor for this state

            :attr:`state` (:attr:`float`): value of the state parameter for this data set

            :attr:`data` (:attr:`array_like`): coordinates to be shifted

        Returns:
            :attr:`transformed` (:attr:`array_like`): the transformed coordinates
        """
        if self.scale == "log":
            transformed = data + np.log(param)
        else:
            transformed = data*param
        return transformed

    def backward(self, param, state, data):
        r"""
        Run a forward shift on the data (from the reference state to the current state).

        Args:
            :attr:`param` (:attr:`float`): value of the shift factor for this state

            :attr:`state` (:attr:`float`): value of the state parameter for this data set

            :attr:`data` (:attr:`array_like`): coordinates to be shifted

        Returns:
            :attr:`transformed` (:attr:`array_like`): the transformed coordinates
        """
        if self.scale == "log":
            transformed = data - np.log(param)
        else:
            transformed = data/param
        return transformed
