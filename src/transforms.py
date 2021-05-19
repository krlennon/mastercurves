import numpy as np

class Multiply():
    """
    Class definition for a multiplicative shift.
    """
    def __init__(self, bounds=(1E-2,1), scale="log"):
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
