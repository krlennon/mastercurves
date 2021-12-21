import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel, ConstantKernel
from scipy import optimize
import pandas as pd
import numdifftools as nd
import random

class MasterCurve:
    """
    Class definition for a master curve, consisting of multiple data sets superimposed.
    """
    def __init__(self):
        """
        Initialize object with some x_data sets and corresponding y_data sets.
        """
        self.xdata = []
        self.xtransformed = []
        self.ydata = []
        self.ytransformed = []
        self.states = []

        # Set the default Gaussian Process kernel
        self.kernel = (RationalQuadratic() * ConstantKernel() + ConstantKernel()
                + WhiteKernel() + WhiteKernel(0.04**2, "fixed"))
        self.gps = []

        # Set the default x and y transformations to the identity
        self.htransforms = []
        self.hparam_names = []
        self.hparams = []
        self.huncertainties = []
        self.hbounds = []
        self.hshared = []
        self.vtransforms = []
        self.vparam_names = []
        self.vparams = []
        self.vuncertainties = []
        self.vbounds = []
        self.vshared = []

    def add_data(self, xdata_new, ydata_new, states_new):
        """
        Add a data set to the master curve.
        Inputs:
            xdata (list) - x-data sets corresponding to one or more states.
                            Each element in the list corresponds to the x-data at a given state.
            ydata (list) - y-data sets corresponding to one or more states.
                            Each element in the list corresponds to the y-data at a given state.
            states (list) - values of the state parameters corresponding to the elements in x_data and y_data.
        """
        self.xdata += xdata_new
        self.ydata += ydata_new
        self.states += states_new

    def clear(self):
        """
        Clear the master curve's data (useful for memory management).
        """
        self.xdata = []
        self.ydata = []
        self.states = []
        self.gps = []
        self.htransforms = []
        self.hparam_names = []
        self.hparams = []
        self.huncertainties = []
        self.hbounds = []
        self.hshared = []
        self.vtransforms = []
        self.vparam_names = []
        self.vparams = []
        self.vuncertainties = []
        self.vbounds = []
        self.vshared = []

    def set_gp_kernel(self, kernel):
        """
        Set the kernel function for the Gaussian Processes used to fit the data.
        Inputs:
            kernel (from sklearn.gaussian_process.kernels) - the (potentially composite) kernel function.
        """
        self.kernel = kernel

    def _fit_gps(self, overwrite=False):
        """
        Fit Gaussian Process models to each data set in this master curve.
        Inputs:
            overwrite (Boolean) - overwrite existing GP fits if any exist (when True).
        """
        # Clear GPs if we are overwriting
        if overwrite:
            self.gps = []

        # Fit GPs in order
        n_fit = len(self.gps)
        for n in range(0,len(self.states)):
            # Don't re-fit GPs if they exist
            if n < n_fit:
                continue
            x = self.xdata[n]
            y = self.ydata[n]
            self.gps += [GaussianProcessRegressor(kernel=self.kernel).fit(x.reshape(-1,1), y)]

    def add_htransform(self, htransform):
        """
        Add a horizontal transformation (or series of transformations) to the master curve (sequantially).
        Inputs:
            htransform - transformation object
        """
        self.htransforms += [htransform]
        if htransform.type == "Multiply":
            self.hparam_names += ["a"]
        else:
            self.hparam_names += [htransform.param]
        self.hparams += [htransform.default]
        self.huncertainties += [0.0]
        self.hbounds += [htransform.bounds]
        self.hshared += [htransform.shared]

    def add_vtransform(self, vtransform):
        """
        Add a vertical transformation (or series of transformations) to the master curve (sequantially).
        Inputs:
            vtransform - transformation object
        """
        self.vtransforms += [vtransform]
        if vtransform.type == "Multiply":
            self.vparam_names += ["b"]
        else:
            self.vparam_names += [transform.param]
        self.vparams += [vtransform.default]
        self.vuncertainties += [0.0]
        self.vbounds += [vtransform.bounds]
        self.vshared += [vtransform.shared]

    def _change_states(self, transforms, data, s1, s2, p1, p2):
        """
        Given a set of transforms, transform the data coordinate from state s1 to state s2.
        Inputs:
            transforms - the list of transforms to take from a state to the reference
            data - the data coordinate vector
            s1, s2 - beginning state (s1) and final state (s2)
            p1, p2 - transforming parameters corresponding to s1 and s2
        Outputs:
            transformed - transformed data from s1 to s2
        """
        transformed = data
        for i in range(len(transforms)):
            transformed = transforms[i].forward(p1[i], s1, transformed)
        for i in range(len(transforms)-1,-1,-1):
            transformed = transforms[i].backward(p2[i], s2, transformed)
        return transformed

    def _unpack(self, ind):
        """
        Unpack the data associated with index ind.
        Inputs:
            ind - index in lists
        Outputs:
            x - x-coordinate data
            y - y-coordinate data
            s - state
            gp - fit GP model
        """
        x = self.xdata[ind]
        y = self.ydata[ind]
        s = self.states[ind]
        gp = self.gps[ind]
        return x, y, s, gp

    def _pairwise_nlog_posterior(self, ind1, ind2, hp1, hp2, vp1, vp2, lamh, lamv):
        """
        Compute the pairwise negative log likelihood loss between two curves.
        Inputs:
            ind1, ind2 - the indices in the data lists corresponding to the two data sets.
            hp1, hp2 - the horizontal shift parameters for each data set
            vp1, vp2 - the vertical shift parameters for each data set
            lamh, lamv - hyperparameter(s) defining the prior distributions over horizontal and vertical shifts
        Outputs:
            loss - the negative log of the posterior, computed over these two data sets.
        """
        # Get the data and GPs
        x1, y1, s1, gp1 = self._unpack(ind1)
        x2, y2, s2, gp2 = self._unpack(ind2)

        # Transform to the other state
        x1s2 = self._change_states(self.htransforms, x1, s1, s2, hp1, hp2)
        x2s1 = self._change_states(self.htransforms, x2, s2, s1, hp2, hp1)

        # Transform to the other state
        y1s2 = self._change_states(self.vtransforms, y1, s1, s2, vp1, vp2)
        y2s1 = self._change_states(self.vtransforms, y2, s2, s1, vp2, vp1)

        # Get the GP predictions
        mu1, sig1 = gp2.predict(x1s2.reshape(-1,1), return_std=True)
        mu2, sig2 = gp1.predict(x2s1.reshape(-1,1), return_std=True)

        # Compute the NLL
        nll1 = (y1s2 - mu1)**2/(2*sig1**2) + np.log(sig1**2)/2 + np.log(2*np.pi)/2
        nll2 = (y2s1 - mu2)**2/(2*sig2**2) + np.log(sig2**2)/2 + np.log(2*np.pi)/2

        # Compute the likelihood
        loss = np.sum(nll1) + np.sum(nll2)

        # Include priors to get the posterior loss
        for i in range(len(self.htransforms)):
            loss += (len(nll1) + len(nll2))*self.htransforms[i].prior(hp1[i]/hp2[i], lamh)
        for i in range(len(self.vtransforms)):
            loss += (len(nll1) + len(nll2))*self.vtransforms[i].prior(vp1[i]/vp2[i], lamv)
        return loss

    def _objective(self, inds, hps, vps, lamh, lamv):
        """
        Compute the loss over all data sets in inds, given params hps and vps.
        Inputs:
            inds - indices of data curves to shift
            hps - horizontal shifting parameters
            vps - vertical shifting parameters
            lamh, lamv - the horizontal and vertical regularization hyperparameters
        Outputs:
            loss - the total loss
        """
        # For every sequential pair in inds
        loss = 0
        vp_store = []
        for i in inds[:-1]:
            ind1 = i
            ind2 = i+1

            # Get the appropriate parameters
            hp1 = []
            hp2 = []
            for k in range(len(hps)):
                if self.hshared[k]:
                    hp1 += list(hps[k])
                    hp2 += list(hps[k])
                else:
                    if ind1 == inds[0]:
                        # Take the first state as a reference for the current batch of shifts
                        hp1 += [self.hparams[k][ind1]]
                    else:
                        hp1 += [hps[k][ind1 - inds[0] - 1]]
                    hp2 += [hps[k][ind2 - inds[0] - 1]]
            vp1 = []
            vp2 = []
            for k in range(len(vps)):
                if self.vshared[k]:
                    vp1 += list(vps[k])
                    vp2 += list(vps[k])
                else:
                    if ind1 == inds[0]:
                        # Take the first state as a reference for the current batch of shifts
                        vp1 += [self.vparams[k][ind1]]
                    else:
                        vp1 += [vps[k][ind1 - inds[0] - 1]]
                    vp2 += [vps[k][ind2 - inds[0] - 1]]

            # Check if the only vertical shift is a Multiply in log-space
            if len(self.vtransforms) == 1:
                if self.vtransforms[0].type == "Multiply":
                    if self.vtransforms[0].scale == "log":
                        # Compute the vertical shift analytically
                        x1, y1, s1, gp1 = self._unpack(ind1)
                        x2, y2, s2, gp2 = self._unpack(ind2)
                        x1s2 = self._change_states(self.htransforms, x1, s1, s2, hp1, hp2)
                        x2s1 = self._change_states(self.htransforms, x2, s2, s1, hp2, hp1)
                        mu1, sig1 = gp2.predict(x1s2.reshape(-1,1), return_std=True)
                        mu2, sig2 = gp1.predict(x2s1.reshape(-1,1), return_std=True)
                        sig_prod = 1/np.sqrt(np.sum(1/(sig1**2)) + np.sum(1/(sig2**2)))
                        mu_prod = (sig_prod**2)*(np.sum((y1 - mu1)/sig1**2) + np.sum((mu2 - y2)/sig2**2))
                        if ind1 == inds[0]:
                            vp1 += [self.vparams[0][ind1]]
                            vp_store += [vp1[0]]
                        else:
                            vp1 += [vp_store[-1]]
                        vp2 += [vp_store[-1]*np.exp(mu_prod)]
                        vp_store += [vp2[-1]]

            # Compute the loss
            loss += self._pairwise_nlog_posterior(ind1, ind2, hp1, hp2, vp1, vp2, lamh, lamv)
        return loss

    def superpose(self, lamh=None, lamv=None):
        """
        Optimize the transformations to superpose the data sets onto a single master curve.
        Inputs:
            lamh, lamv - regularization hyperparameters for vertical and horizontal shifts
        Outputs:
            loss_min - the minimum loss computed during superposition (either total, or pairwise)
        """
        # First, fit the GPs
        self._fit_gps()

        # First, determine if any transformations have shared parameters
        pairwise = True
        if True in self.hshared or True in self.vshared:
            pairwise = False

        nsets = len(self.states)
        if not pairwise:
            n = nsets
        else:
            n = 2

        # Initialize the search
        hparams = [0]
        bounds = []
        k = 0
        for htransform in self.htransforms:
            if htransform.shared:
                hparams += [hparams[-1] + 1]
                bounds += [htransform.bounds]
                hpval = self.hparams[k]
                self.hparams[k] = [hpval]
                self.huncertainties[k] = [0.0]
            else:
                hparams += [hparams[-1] + n - 1]
                bounds += (n - 1)*[htransform.bounds]
                hpval = self.hparams[k]
                self.hparams[k] = nsets*[hpval]
                self.huncertainties[k] = nsets*[0.0]
            k += 1
        vparams = [hparams[-1]]
        k = 0
        for vtransform in self.vtransforms:
            if vtransform.type == "Multiply" and len(self.vtransforms) == 1:
                if vtransform.scale == "log":
                    vpval = self.vparams[k]
                    self.vparams[k] = nsets*[vpval]
                    self.vuncertainties[k] = nsets*[0.0]
                    continue
            if vtransform.shared:
                vparams += [vparams[-1] + 1]
                bounds += [vtransform.bounds]
                vpval = self.vparams[k]
                self.vparams[k] = [vpval]
                self.vuncertainties[k] = [0.0]
            else:
                vparams += [vparams[-1] + n - 1]
                bounds += (n - 1)*[vtransform.bounds]
                vpval = self.vparams[k]
                self.vparams[k] = nsets*[vpval]
                self.vuncertainties[k] = nsets*[0.0]
            k += 1

        # Set up the objective for simultaneous shifting of all data sets
        if not pairwise:
            inds = [i for i in range(nsets)]
            # Check if hyperparameters are present (hyperparameters not currently supported for non-pairwise)
            if lamh == None:
                lamh_val = 0
            if lamv == None:
                lamv_val = 0

            # Set up the objective
            obj = lambda p: self._objective(inds, [p[hparams[i-1]:hparams[i]] for i in range(1,len(hparams))],
                    [p[vparams[i-1]:vparams[i]] for i in range(1,len(vparams))], lamh_val, lamv_val)

            # If there's only one parameter to optimize over, then do a linesearch
            if vparams[-1] == 1:
                pv = np.linspace(bounds[0][0], bounds[0][1], 100)
                losses = []
                for p in pv:
                    losses += [obj([p])]
                losses = np.array(losses)
                ind_min = np.argmin(losses)
                loss_min = losses[ind_min]
                p_min = pv[ind_min]

                # Gradient descent around the global min of the linesearch
                if ind_min == 0:
                    bds = [(pv[0],pv[1])]
                elif ind_min == len(pv)-1:
                    bds = [(pv[-2],pv[-1])]
                else:
                    bds = [(pv[ind_min-1],pv[ind_min+1])]
                res = optimize.minimize(obj, [p_min], bounds=bds)
                p_min = res.x[0]
                loss_min = res.fun
                hess = nd.Hessian(obj, step=p_min/10)(res.x)

                # Assign the optimal parameter
                for k in range(len(hparams[1:])):
                    if hparams[k+1] - hparams[k] > 0:
                        self.hparams[k] = [p_min]
                        self.huncertainties[k] = [1/np.sqrt(hess[0][0])]
                for k in range(len(vparams[1:])):
                    if vparams[k+1] - vparams[k] > 0:
                        self.vparams[k] = [p_min]
                        self.vuncertainties[k] = [1/np.sqrt(hess[0][0])]

                # If necessary, compute the vertical shifts analytically
                if len(self.vtransforms) == 1:
                    if self.vtransforms[0].type == "Multiply":
                        if self.vtransforms[0].scale == "log":
                            for i in range(nsets-1):
                                # Compute the vertical shift analytically
                                ind1 = i
                                ind2 = i+1
                                x1, y1, s1, gp1 = self._unpack(ind1)
                                x2, y2, s2, gp2 = self._unpack(ind2)
                                x1s2 = self._change_states(self.htransforms, x1, s1, s2, [p_min], [p_min])
                                x2s1 = self._change_states(self.htransforms, x2, s2, s1, [p_min], [p_min])
                                mu1, sig1 = gp2.predict(x1s2.reshape(-1,1), return_std=True)
                                mu2, sig2 = gp1.predict(x2s1.reshape(-1,1), return_std=True)
                                sig_prod = 1/np.sqrt(np.sum(1/(sig1**2)) + np.sum(1/(sig2**2)))
                                mu_prod = (sig_prod**2)*(np.sum((y1 - mu1)/sig1**2) + np.sum((mu2 - y2)/sig2**2))
                                vp1 = self.vparams[0][ind1]
                                vp2 = vp1*np.exp(mu_prod)
                                self.vparams[0][ind2] = vp2
        else:
            # Perform pairwise shifting
            loss_min = []
            self.uncertainties = []
            for m in range(nsets-1):
                inds = [m, m+1]

                # Check for hyperarameters
                if lamh == None:
                    lamh_val = 0
                else:
                    lamh_val = lamh[m]
                if lamv == None:
                    lamv_val = 0
                else:
                    lamv_val = lamv[m]

                # Set up the objective
                obj = lambda p: self._objective(inds, [p[hparams[i-1]:hparams[i]] for i in range(1,len(hparams))],
                        [p[vparams[i-1]:vparams[i]] for i in range(1,len(vparams))], lamh_val, lamv_val)

                # If there's only one parameter to optimize over, then do a linesearch
                if vparams[-1] == 1:
                    # Shift so that ind1 is the new reference state
                    pv = np.logspace(np.log10(bounds[0][0]), np.log10(bounds[0][1]), 100)
                    for k in range(len(hparams[1:])):
                        if hparams[k+1] - hparams[k] > 0:
                            pv = pv*self.hparams[k][m]

                    # Linesearch
                    losses = []
                    for p in pv:
                        losses += [obj([p])]
                    losses = np.array(losses)
                    ind_min = np.argmin(losses)
                    p_min = pv[ind_min]

                    # Gradient descent around the global min of the linesearch
                    if ind_min == 0:
                        bds = [(pv[0],pv[1])]
                    elif ind_min == len(pv)-1:
                        bds = [(pv[-2],pv[-1])]
                    else:
                        bds = [(pv[ind_min-1],pv[ind_min+1])]
                    res = optimize.minimize(obj, [p_min], bounds=bds)
                    p_min = res.x[0]
                    loss_min += [res.fun]
                    try:
                        hess = nd.Hessian(obj, step=p_min/10)(res.x)
                    except IndexError:
                        # NumDiffTools failed, probably because p_min is at the edge of the search range
                        dp = p_min/10
                        hess = [[(obj([p_min + dp]) - 2*obj([p_min]) + obj([p_min - dp]))/(dp**2)]]

                    # Assign the optimal parameter
                    for k in range(len(hparams[1:])):
                        if hparams[k+1] - hparams[k] > 0:
                            self.hparams[k][m+1] = p_min
                            self.huncertainties[k][m+1] = 1/np.sqrt(hess[0][0])
                    for k in range(len(vparams[1:])):
                        if vparams[k+1] - vparams[k] > 0:
                            self.vparams[k][m+1] = p_min
                            self.vuncertainties[k][m+1] = 1/np.sqrt(hess[0][0])

                    # If necessary, compute the vertical shift analytically
                    if len(self.vtransforms) == 1:
                        if self.vtransforms[0].type == "Multiply":
                            if self.vtransforms[0].scale == "log":
                                ind1 = inds[0]
                                ind2 = inds[1]
                                x1, y1, s1, gp1 = self._unpack(ind1)
                                x2, y2, s2, gp2 = self._unpack(ind2)
                                x1s2 = self._change_states(self.htransforms, x1, s1, s2, [self.hparams[0][ind1]], [p_min])
                                x2s1 = self._change_states(self.htransforms, x2, s2, s1, [p_min], [self.hparams[0][ind1]])
                                mu1, sig1 = gp2.predict(x1s2.reshape(-1,1), return_std=True)
                                mu2, sig2 = gp1.predict(x2s1.reshape(-1,1), return_std=True)
                                sig_prod = 1/np.sqrt(np.sum(1/(sig1**2)) + np.sum(1/(sig2**2)))
                                mu_prod = (sig_prod**2)*(np.sum((y1 - mu1)/sig1**2) + np.sum((mu2 - y2)/sig2**2))
                                vp1 = self.vparams[0][ind1]
                                vp2 = vp1*np.exp(mu_prod)
                                self.vparams[0][ind2] = vp2
                                self.vuncertainties[0][ind2] = vp1*np.exp(mu_prod)*np.sqrt(sig_prod)
                if vparams[-1] == 0:
                    # If there are no parameters to shift, check if we should still compute vertical shifts
                    if len(self.vtransforms) == 1:
                        if self.vtransforms[0].type == "Multiply":
                            if self.vtransforms[0].scale == "log":
                                ind1 = inds[0]
                                ind2 = inds[1]
                                x1, y1, s1, gp1 = self._unpack(ind1)
                                x2, y2, s2, gp2 = self._unpack(ind2)
                                x1s2 = self._change_states(self.htransforms, x1, s1, s2, [], [])
                                x2s1 = self._change_states(self.htransforms, x2, s2, s1, [], [])
                                mu1, sig1 = gp2.predict(x1s2.reshape(-1,1), return_std=True)
                                mu2, sig2 = gp1.predict(x2s1.reshape(-1,1), return_std=True)
                                sig_prod = 1/np.sqrt(np.sum(1/(sig1**2)) + np.sum(1/(sig2**2)))
                                mu_prod = (sig_prod**2)*(np.sum((y1 - mu1)/sig1**2) + np.sum((mu2 - y2)/sig2**2))
                                vp1 = self.vparams[0][ind1]
                                vp2 = vp1*np.exp(mu_prod)
                                self.vparams[0][ind2] = vp2

        # Return error if more than one parameter is specified (other than vertical log Multiply)
        if vparams[-1] > 1:
            raise ValueError("Too many transformations provided. Current version supports one tranformation (in addition to one vertical log Multiply())")

        self._shift_data()
        return loss_min

    def _shift_data(self):
        """
        Save the shifted data for this master curve.
        """
        self.xtransformed = []
        self.ytransformed = []
        for k in range(len(self.xdata)):
            x = self.xdata[k]
            for l in range(len(self.htransforms)):
                if self.hshared[l]:
                    p = self.hparams[l][0]
                else:
                    p = self.hparams[l][k]
                x = self.htransforms[l].forward(p, self.states[k], x)
            self.xtransformed += [x]
            y = self.ydata[k]
            for l in range(len(self.vtransforms)):
                if self.vshared[l]:
                    p = self.vparams[l][0]
                else:
                    p = self.vparams[l][k]
                y = self.vtransforms[l].forward(p, self.states[k], y)
            self.ytransformed += [y]

    def plot(self, log=True, colormap=plt.cm.tab10, colorby="index"):
        """
        Plot the data, GPs, and master curve.
        Inputs:
            log - if True, the data is the log of the desired quantity, so plot in linear space
            colormap - colormap for plotting curves with different states.
        """
        # Plot the data
        fig1, ax1 = plt.subplots(1,1)

        # Plot the data with the GPs
        fig2, ax2 = plt.subplots(1,1)

        if colorby == "state":
            colorfn = lambda k: (self.states[k] - np.min(self.states))/(np.max(self.states) - np.min(self.states))
        else:
            colorfn = lambda k: k/len(self.states)

        for k in range(len(self.xdata)):
            if log:
                ax1.loglog(np.exp(self.xdata[k]), np.exp(self.ydata[k]), 'o', label=str(self.states[k]),
                        color=colormap(colorfn(k)))
                ax2.loglog(np.exp(self.xdata[k]), np.exp(self.ydata[k]), 'o', label=str(self.states[k]),
                        color=colormap(colorfn(k)))
            else:
                ax1.plot(self.xdata[k], self.ydata[k], 'o', label=str(self.states[k]),
                        color=colormap(colorfn(k)))
                ax2.plot(self.xdata[k], self.ydata[k], 'o', label=str(self.states[k]),
                        color=colormap(colorfn(k)))
        ax1.legend()

        # Add the GPs to the second figure
        xlim = ax2.get_xlim()
        if log:
            xgp = np.linspace(np.log(xlim[0]),np.log(xlim[1]),100)
        else:
            xgp = np.linspace(xlim[0],xlim[1],100)
        for k in range(len(self.gps)):
            y, s = self.gps[k].predict(xgp.reshape(-1,1), return_std=True)
            if log:
                ax2.loglog(np.exp(xgp), np.exp(y), color=colormap(colorfn(k)))
                ax2.fill_between(np.exp(xgp), np.exp(y - s), np.exp(y + s), color=colormap(colorfn(k)), alpha=0.2)
            else:
                ax2.plot(xgp, y, color=colormap(colorfn(k)))
                ax2.fill_between(xgp, y - s, y + s, color=colormap(colorfn(k)), alpha=0.2)
        ax2.set_xlim(xlim)

        # Plot the master curve
        fig3, ax3 = plt.subplots(1,1)

        for k in range(len(self.xtransformed)):
            if log:
                ax3.loglog(np.exp(self.xtransformed[k]), np.exp(self.ytransformed[k]), 'o', label=str(self.states[k]),
                        color=colormap(colorfn(k)))
            else:
                ax3.plot(self.xtransformed[k], self.ytransformed[k], 'o', label=str(self.states[k]),
                        color=colormap(colorfn(k)))

        return fig1, ax1, fig2, ax2, fig3, ax3

    def change_ref(self, ref_state, a_ref=1, b_ref=1):
        """
        Change the reference state.
        Inputs:
            ref_state - the new reference state, which may be one of the current states or not
            a_ref - if ref_state is not a current state, must be provided. This is the new reference's horizontal shift
                    with respect to the current reference
            b_ref - if ref_state is not a current state, must be provided. This is the new reference's vertical shift
                    with respect to the current reference
        """
        if len(self.htransforms) == 1:
            if self.htransforms[0].type == "Multiply":
                if ref_state in self.states and a_ref == 1:
                    a_ref = self.hparams[0][self.states.index(ref_state)]
                for k in range(len(self.hparams[0])):
                    self.hparams[0][k] = self.hparams[0][k]/a_ref
        if len(self.vtransforms) == 1:
            if self.vtransforms[0].type == "Multiply":
                if ref_state in self.states and b_ref == 1:
                    b_ref = self.vparams[0][self.states.index(ref_state)]
                for k in range(len(self.vparams[0])):
                    self.vparams[0][k] = self.vparams[0][k]/b_ref
        self._shift_data()

    def output_table(self, file=None):
        """
        Write a csv file with a table of all parameters (or return a data frame if no file given).
        Inputs:
            file - the (optional) file to which the data frame is written
        """
        data = {"state": self.states}
        for i in range(len(self.htransforms)):
            if self.hshared[i]:
                data[self.hparam_names[i]] = len(self.states)*self.hparams[i]
            else:
                data[self.hparam_names[i]] = self.hparams[i]
        for i in range(len(self.vtransforms)):
            if self.vshared[i]:
                data[self.vparam_names[i]] = len(self.states)*self.vparams[i]
            else:
                data[self.vparam_names[i]] = self.vparams[i]
        df = pd.DataFrame(data=data)

        # Write to file or just return
        if file != None:
            df.to_csv(file, index=None)
        return df

    def _mccv(self, x1, y1, s1, x2, y2, s2, lamh, lamv, alpha, folds):
        """
        Compute the score of a Monte-Carlo cross validation test.
        Inputs:
            x1, y1, s1 - the first data set
            x2, y2, s2 - the second data set
            lamh, lamv - horizontal and vertical hyperparameters for the prior distribution
            alpha - the fraction of points kept in the training set
            folds - the number of MCCV folds to average over
        Outputs:
            score - the MCCV score (averaged over folds)
        """
        scores = []
        for f in range(folds):
            # Monte Carlo sampling of the data with alpha
            ind1train = random.sample(range(len(x1)), int(np.floor(len(x1)*alpha)))
            ind2train = random.sample(range(len(x2)), int(np.floor(len(x2)*alpha)))
            ind1val = list(set(range(len(x1))) - set(ind1train))
            ind2val = list(set(range(len(x2))) - set(ind2train))
            x1val = x1[ind1val]
            y1val = y1[ind1val]
            x2val = x2[ind2val]
            y2val = y2[ind2val]
            x1train = x1[ind1train]
            y1train = y1[ind1train]
            x2train = x2[ind2train]
            y2train = y2[ind2train]

            # Make a small master curve for these two data sets
            mc = MasterCurve()
            mc.add_data([x1train, x2train], [y1train, y2train], [s1, s2])
            mc.set_gp_kernel(self.kernel)

            # Add the transformations
            for htransform in self.htransforms:
                mc.add_htransform(htransform)
            for vtransform in self.vtransforms:
                mc.add_vtransform(vtransform)

            # Superpose
            mc.superpose([lamh], [lamv])

            # Get the transformed data and fit new GPs
            xtransformed = np.concatenate(mc.xtransformed)
            ytransformed = np.concatenate(mc.ytransformed)
            gptransformed = GaussianProcessRegressor(kernel=self.kernel).fit(xtransformed.reshape(-1,1),
                    ytransformed)

            # Transform the validation set to the reference state
            x1valtransformed = x1val
            x2valtransformed = x2val
            for l in range(len(mc.htransforms)):
                if mc.hshared[l]:
                    p1 = mc.hparams[l][0]
                    p2 = mc.hparams[l][0]
                else:
                    p1 = mc.hparams[l][0]
                    p2 = mc.hparams[l][1]
                x1valtransformed = mc.htransforms[l].forward(p1, s1, x1valtransformed)
                x2valtransformed = mc.htransforms[l].forward(p2, s2, x2valtransformed)
            y1valtransformed = y1val
            y2valtransformed = y2val
            for l in range(len(mc.vtransforms)):
                if mc.vshared[l]:
                    p1 = mc.vparams[l][0]
                    p2 = mc.vparams[l][0]
                else:
                    p1 = mc.vparams[l][0]
                    p2 = mc.vparams[l][1]
                y1valtransformed = mc.vtransforms[l].forward(p1, s1, y1valtransformed)
                y2valtransformed = mc.vtransforms[l].forward(p2, s2, y2valtransformed)

            # Get predictions from the shifted GPs
            y1pred, sig1 = gptransformed.predict(x1valtransformed.reshape(-1,1), return_std=True)
            y2pred, sig2 = gptransformed.predict(x2valtransformed.reshape(-1,1), return_std=True)

            # Compute the MSE
            nll1 = (y1valtransformed - y1pred)**2/(2*sig1**2) + np.log(sig1**2)/2 + np.log(2*np.pi)/2
            nll2 = (y2valtransformed - y2pred)**2/(2*sig2**2) + np.log(sig2**2)/2 + np.log(2*np.pi)/2
            scores += [np.sum(nll1) + np.sum(nll2)]

        # Compute average and standard error
        scores = np.array(scores)
        average = np.sum(scores)/folds
        sterr = np.sqrt(np.sum((scores - average)**2))/folds
        return average, sterr

    def hpopt(self, lamh=None, lamv=None, npoints=100, alpha=0.5, folds=10):
        """
        Perform hyperparameter optimization on the prior (regularization) using MCCV.
        Inputs:
            lamh, lamv - ranges (tuples) for the hyperparameter search. If not searching this parameter, then None
            npoints - number of grid points to search
            alpha - keep rate for MCCV
            folds - number of MCCV folds
        Outputs:
            lamh_opt, lamv_opt - optimal hyperparameters
        """
        # Loop through each pair of data sets (only pairwise optimization supported)
        lamh_mins = []
        lamv_mins = []
        for m in range(len(self.states)-1):
            x1 = self.xdata[m]
            y1 = self.ydata[m]
            s1 = self.states[m]
            x2 = self.xdata[m+1]
            y2 = self.ydata[m+1]
            s2 = self.states[m+1]

            # Check which parameter to optimize over
            if lamh == None:
                lamh_val = 0.0
                scores = []
                sterrs = []
                lamv_v = np.logspace(np.log10(lamv[0]),np.log10(lamv[1]),npoints)
                #lamv_v = np.linspace((lamv[0]),(lamv[1]),npoints)
                for lamv_val in lamv_v:
                    average, sterr = self._mccv(x1, y1, s1, x2, y2, s2, lamh_val, lamv_val, alpha, folds)
                    scores += [average]
                    sterrs += [sterr]
                ind_min = np.argmin(scores)
                score_min = scores[ind_min]
                sterr_min = sterrs[ind_min]

                # Use the one standard error method to find the optimal lambda
                lamv_val = np.max(np.where(np.array(scores) <= score_min + sterr_min, lamv_v, np.zeros(len(lamv_v))))
                lamv_mins += [lamv_val]
                lamh_mins += [0.0]
            elif lamv == None:
                lamv_val = 0.0
                scores = []
                sterrs = []
                lamh_v = np.logspace(np.log10(lamh[0]),np.log10(lamh[1]),npoints)
                #lamh_v = np.linspace((lamh[0]),(lamh[1]),npoints)
                for lamh_val in lamh_v:
                    average, sterr = self._mccv(x1, y1, s1, x2, y2, s2, lamh_val, lamv_val, alpha, folds)
                    scores += [average]
                    sterrs += [sterr]
                ind_min = np.argmin(scores)
                score_min = scores[ind_min]
                sterr_min = sterrs[ind_min]

                # Use the one standard error method to find the optimal lambda
                lamh_val = np.max(np.where(np.array(scores) <= score_min + sterr_min, lamh_v, np.zeros(len(lamh_v))))
                lamh_mins += [lamh_val]
                lamv_mins += [0.0]
        return lamh_mins, lamv_mins
