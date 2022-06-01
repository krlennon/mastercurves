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
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel, ConstantKernel
from scipy import optimize
import pandas as pd
import numdifftools as nd
import random

class MasterCurve:
    r"""
    Class definition for a master curve, consisting of multiple data sets superimposed.

    A :attr:`MasterCurve` object will contain all of the data used to construct a master
    curve, along with the coordinate transformations, parameters for coordinate transformations
    (such as shift factors), associated uncertainties, Gaussian process models, and
    transformed data.

    Attributes:
        :attr:`xdata` (:attr:`list[array_like]`): list whose elements are the independent
        coordinates for a data set at a given state

        :attr:`ydata` (:attr:`list[array_like]`): list whose elements are the dependent
        coordinates for a data set at a given state

        :attr:`states` (:attr:`list[float]`): list whose elements are the value of the state
        defining each data set

        :attr:`htransforms` (:attr:`list[Transform]`): list whose elements are the
        coordinate transforms performed on the independent coordinate (i.e. horizontally)

        :attr:`hparam_names` (:attr:`list[string]`): list whose elements are the names
        of the parameters for the horizontal transforms

        :attr:`hbounds` (:attr:`list[tuple]`): list whose elements are the upper and lower
        bounds for the horizontal transformation parameters

        :attr:`hshared` (:attr:`list[bool]`): list whose elements indicate whether the
        corresponding element of :attr:`hparam_names` is a parameter whose value is shared
        across all states (:attr:`True`) or takes on an independent value for each state
        (:attr:`False`)

        :attr:`hparams` (:attr:`list[list[float]]`): list whose elements are the values of the
        horizontal transformation parameters at each state

        :attr:`huncertainties` (:attr:`list[float]`): list whose elements are the uncertainties
        associated with the values in :attr:`hparams`

        :attr:`vtransforms` (:attr:`list[Transform]`): list whose elements are the
        coordinate transforms performed on the dependent coordinate (i.e. vertically)

        :attr:`vparam_names` (:attr:`list[string]`): list whose elements are the names
        of the parameters for the vertical transforms

        :attr:`vbounds` (:attr:`list[tuple]`): list whose elements are the upper and lower
        bounds for the vertical transformation parameters

        :attr:`vshared` (:attr:`list[bool]`): list whose elements indicate whether the
        corresponding element of :attr:`vparam_names` is a parameter whose value is shared
        across all states (:attr:`True`) or takes on an independent value for each state
        (:attr:`False`)

        :attr:`vparams` (:attr:`list[list[float]]`): list whose elements are the values of the
        vertical transformation parameters at each state

        :attr:`vuncertainties` (:attr:`list[float]`): list whose elements are the uncertainties
        associated with the values in :attr:`vparams`

        :attr:`kernel` (:attr:`sklearn.gaussian_process.kernels.Kernel`): kernel function
        for the Gaussian process model

        :attr:`gps` (:attr:`list[sklearn.gaussian_process.GaussianProcessRegressor]`): list
        whose elements are Gaussian process models for each state

        :attr:`xtransformed` (:attr:`list[array_like]`): list whose elements are the transformed
        independent coordinates for a data set at a given state

        :attr:`ytransformed` (:attr:`list[array_like]`): list whose elements are the transformed
        dependent coordinates for a data set at a given state
    """
    def __init__(self, fixed_noise=0.04):
        r"""
        Initialize a MasterCurve object.

        Args:
            :attr:`fixed_noise` (:attr:`float`): the fixed noise level for the Gaussian process
            models, corresponding to experimental uncertainty not evident in the data. By default,
            the noise level is 0.04.
        """
        self.xdata = []
        self.xtransformed = []
        self.ydata = []
        self.ytransformed = []
        self.states = []

        # Set the default Gaussian Process kernel
        self.kernel = (RationalQuadratic() * ConstantKernel() + ConstantKernel()
                + WhiteKernel() + WhiteKernel(fixed_noise**2, "fixed"))
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
        r"""
        Add a data set or data sets to the master curve.

        Args:
            :attr:`xdata_new` (:attr:`array_like` or :attr:`list[array_like]`): array(s) corresponding to
            the dependent coordinates for given states

            :attr:`ydata_new` (:attr:`array_like` or :attr:`list[array_like]`): array(s) corresponding to
            the independent coordinates for given states

            :attr:`states_new` (:attr:`float` or :attr:`list[float]`): values of the state parameter
            corresponding to the data in :attr:`xdata_new` and :attr:`ydata_new`
        """
        self.xdata += xdata_new
        self.ydata += ydata_new
        self.states += states_new

    def clear(self):
        r"""
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
        r"""
        Set the kernel function for the Gaussian Processes used to fit the data.

        Args:
            :attr:`kernel` (:attr:`sklearn.gaussian_process.kernels.Kernel`): the
            (potentially composite) kernel function
        """
        self.kernel = kernel

    def _fit_gps(self, overwrite=False):
        r"""
        Fit Gaussian Process models to each data set in this master curve.

        Args:
            :attr:`overwrite` (:attr:`bool`): overwrite existing GP fits if any exist (when :attr:`True`).
            By defailt, it is :attr:`False`.
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
        r"""
        Add a horizontal transformation (or series of sequential transformations) to the master curve.

        Args:
            :attr:`htransform` (:attr:`Transform`): object of a Transform class, which implements the
            coordinate transformation and stores information about transformation parameters
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
        r"""
        Add a vertical transformation (or series of sequential transformations) to the master curve.

        Args:
            :attr:`vtransform` (:attr:`Transform`): object of a Transform class, which implements the
            coordinate transformation and stores information about transformation parameters
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
        r"""
        Given a set of transforms, transform the data coordinate from state :attr:`s1` to state :attr:`s2`.

        Args:
            :attr:`transforms` (:attr:`list[Transform]`): the list of transforms to take data
            from state :attr:`s1` to state :attr:`s2`

            :attr:`data` (:attr:`array_like`): vector containing the coordinates to be transformed

            :attr`s1` (:attr:`float`): the initial state

            :attr`s2` (:attr:`float`): the final state

            :attr:`p1` (:attr:list[`float`]): the values of the transform parameters corresponding
            to each transform in :attr:`transforms`, for state :attr:`s1`

            :attr:`p2` (:attr:list[`float`]): the values of the transform parameters corresponding
            to each transform in :attr:`transforms`, for state :attr:`s2`

        Returns:
            :attr:`transformed` (:attr:`array_like`): vector containing the transformed coordinate
        """
        transformed = data
        for i in range(len(transforms)):
            transformed = transforms[i].forward(p1[i], s1, transformed)
        for i in range(len(transforms)-1,-1,-1):
            transformed = transforms[i].backward(p2[i], s2, transformed)
        return transformed

    def _unpack(self, ind):
        r"""
        Unpack the data associated with index :attr:`ind`.

        Args:
            :attr:`ind` (:attr:`int`): index of the state

        Returns:
            :attr:`x` (:attr:`array_like`): corresponding x-coordinates

            :attr:`y` (:attr:`array_like`): corresponding y-coordinates

            :attr:`s` (:attr:`float`): corresponding state

            :attr:`gp` (:attr:`sklearn.gaussian_process.GaussianProcessRegressor): corresponding GP model
        """
        x = self.xdata[ind]
        y = self.ydata[ind]
        s = self.states[ind]
        gp = self.gps[ind]
        return x, y, s, gp

    def _pairwise_nlog_posterior(self, ind1, ind2, hp1, hp2, vp1, vp2, lamh, lamv):
        r"""
        Compute the pairwise negative log posterior loss between two curves.

        Args:
            :attr:`ind1`, :attr:`ind2` (:attr:`int`): the indices in the data lists
            corresponding to the two data sets

            :attr:`hp1`, :attr`hp2` (:attr:`list[float]`): the horizontal shift parameters
            for each data set, corresponding to each coordinate transformation

            :attr:`vp1`, :attr`vp2` (:attr:`list[float]`): the vertical shift parameters
            for each data set, corresponding to each coordinate transformation

            :attr:`lamh`, :attr:`lamv` (:attr:`list[float]`): hyperparameter(s) defining
            the prior distributions over the corresponding horizontal and vertical shifts

        Returns:
            :attr:`loss` (:attr:`float`): the negative log of the posterior,
            computed over these two data sets
        """
        # Get the data and GPs
        x1, y1, s1, gp1 = self._unpack(ind1)
        x2, y2, s2, gp2 = self._unpack(ind2)

        # Transform to the other state
        x1s2 = self._change_states(self.htransforms, x1, s1, s2, hp1, hp2)
        x1s2 = x1s2[~np.isnan(x1s2)]
        x2s1 = self._change_states(self.htransforms, x2, s2, s1, hp2, hp1)
        x2s1 = x2s1[~np.isnan(x2s1)]

        # Transform to the other state
        y1s2 = self._change_states(self.vtransforms, y1, s1, s2, vp1, vp2)
        y2s1 = self._change_states(self.vtransforms, y2, s2, s1, vp2, vp1)

        # Get the GP predictions
        mu1, sig1 = gp2.predict(x1s2.reshape(-1,1), return_std=True)
        mu2, sig2 = gp1.predict(x2s1.reshape(-1,1), return_std=True)

        # Compute the NLL
        nll1 = (y1s2[:len(mu1)] - mu1)**2/(2*sig1**2) + np.log(sig1**2)/2 + np.log(2*np.pi)/2
        nll2 = (y2s1[:len(mu2)] - mu2)**2/(2*sig2**2) + np.log(sig2**2)/2 + np.log(2*np.pi)/2

        # Compute the likelihood
        loss = np.sum(nll1) + np.sum(nll2)

        # Include priors to get the posterior loss
        for i in range(len(self.htransforms)):
            loss += (len(nll1) + len(nll2))*self.htransforms[i].prior(hp1[i]/hp2[i], lamh)
        for i in range(len(self.vtransforms)):
            loss += (len(nll1) + len(nll2))*self.vtransforms[i].prior(vp1[i]/vp2[i], lamv)
        return loss

    def _objective(self, inds, hps, vps, lamh, lamv):
        r"""
        Compute the loss over all data sets in :attr:`inds`, given params :attr:`hps` and :attr:`vps`.

        Args:
            :attr:`inds` (:attr:`list[int]`): indices of data sets to shift

            :attr:`hps` - (:attr:`list[float]` or :attr:`list[list[float]]`): horizontal
            shifting parameters for each state in :attr:`inds` and each coordinate transformation

            :attr:`vps` - (:attr:`list[float]` or :attr:`list[list[float]]`): vertical
            shifting parameters for each state in :attr:`inds` and each coordinate transformation

            :attr:`lamh`, :attr:`lamv` (:attr:`list[float]`) the horizontal and vertical
            regularization hyperparameters

        Returns:
            :attr:`loss` (:attr:`float`): the total loss
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
                        x1s2 = x1s2[~np.isnan(x1s2)]
                        x2s1 = self._change_states(self.htransforms, x2, s2, s1, hp2, hp1)
                        x2s1 = x2s1[~np.isnan(x2s1)]
                        mu1, sig1 = gp2.predict(x1s2.reshape(-1,1), return_std=True)
                        mu2, sig2 = gp1.predict(x2s1.reshape(-1,1), return_std=True)
                        sig_prod = 1/np.sqrt(np.sum(1/(sig1**2)) + np.sum(1/(sig2**2)))
                        mu_prod = (sig_prod**2)*(np.sum((y1[:len(mu1)] - mu1)/sig1**2)
                                + np.sum((mu2 - y2[:len(mu2)])/sig2**2))
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
        r"""
        Optimize the transformations to superpose the data sets onto a single master curve.

        Args:
            :attr:`lamh` (:attr:`list[float]`): hyperparameters for the horizontal shifts,
            corresponding to each state in the master curve. Defaults to :attr:`None`, meaning
            a uniform (unregularized) prior.

            :attr:`lamv` (:attr:`list[float]`): hyperparameters for the vertical shifts,
            corresponding to each state in the master curve. Defaults to :attr:`None`, meaning
            a uniform (unregularized) prior.

        Returns:
            :attr:`loss_min` (:attr:`float`): the minimum loss computed during superposition
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
                                x1s2 = x1s2[~np.isnan(x1s2)]
                                x2s1 = self._change_states(self.htransforms, x2, s2, s1, [p_min], [p_min])
                                x2s1 = x2s1[~np.isnan(x2s1)]
                                mu1, sig1 = gp2.predict(x1s2.reshape(-1,1), return_std=True)
                                mu2, sig2 = gp1.predict(x2s1.reshape(-1,1), return_std=True)
                                sig_prod = 1/np.sqrt(np.sum(1/(sig1**2)) + np.sum(1/(sig2**2)))
                                mu_prod = (sig_prod**2)*(np.sum((y1[:len(mu1)] - mu1)/sig1**2)
                                        + np.sum((mu2 - y2[:len(mu2)])/sig2**2))
                                vp1 = self.vparams[0][ind1]
                                vp2 = vp1*np.exp(mu_prod)
                                self.vparams[0][ind2] = vp2
                                self.vuncertainties[0][ind2] = vp1*np.exp(mu_prod)*np.sqrt(sig_prod)
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
                                self.vuncertainties[0][ind2] = vp1*np.exp(mu_prod)*np.sqrt(sig_prod)

        # Return error if more than one parameter is specified (other than vertical log Multiply)
        if vparams[-1] > 1:
            raise ValueError("Too many transformations provided. Current version supports one tranformation (in addition to one vertical log Multiply())")

        self._shift_data()
        self._propagate_uncertainty()
        return loss_min

    def _shift_data(self):
        r"""
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

    def _propagate_uncertainty(self):
        r"""
        For Multiply transforms, propagate uncertainty estimates in the shift factors
        so that the uncertainties are cumulative.
        """
        # Only supported when there is one coordinate transform per axis
        if len(self.htransforms) == 1:
            if self.htransforms[0].type == "Multiply":
                huncertainties_new = [0]
                for i in range(1, len(self.huncertainties[0])):
                    huncertainties_new += [self.hparams[0][i] *np.sqrt((self.huncertainties[0][i]/self.hparams[0][i])**2 +
                                (huncertainties_new[i-1]/self.hparams[0][i-1])**2)]
                self.huncertainties[0] = huncertainties_new

        # Now vertical uncertainties
        if len(self.vtransforms) == 1:
            if self.vtransforms[0].type == "Multiply":
                vuncertainties_new = [0]
                for i in range(1, len(self.vuncertainties[0])):
                    vuncertainties_new += [self.vparams[0][i] *np.sqrt((self.vuncertainties[0][i]/self.vparams[0][i])**2 +
                                (vuncertainties_new[i-1]/self.vparams[0][i-1])**2)]
                self.vuncertainties[0] = vuncertainties_new

    def plot(self, log=True, colormap=plt.cm.tab10, colorby="index"):
        r"""
        Plot the data, GPs, and master curve.

        Args:
            :attr:`log` (:attr:`bool`): whether the data represents the logarithm of the
            measured quantity. Defaults to :attr:`True`.

            :attr:`colormap` (:attr:`matplotlib.colors.Colormap`): colormap for plotting.
            Defaults to the tab10 colormap.

            :attr:`colorby` (:attr:`string`): how to color the data. Options are :attr:`index`
            for coloring by index, or :attr:`state` for coloring by the value of the state.
            Defaults to :attr:`index`.

        Returns:
            :attr:`fig1`, :attr:`ax1` (:attr:`matplotlib.Figure` and :attr:`matplotlib.axes.Axes`): the
            figure and axes objects displaying the raw (untransformed data)

            :attr:`fig2`, :attr:`ax2` (:attr:`matplotlib.Figure` and :attr:`matplotlib.axes.Axes`): the
            figure and axes objects displaying the untransformed data and GP models

            :attr:`fig3`, :attr:`ax3` (:attr:`matplotlib.Figure` and :attr:`matplotlib.axes.Axes`): the
            figure and axes objects displaying the superposed data (i.e. the master curve)
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
        r"""
        Change the reference state for the master curve.

        Args:
            :attr:`ref_state` (:attr:`float`): the new reference state, which may or may not
            be one of the current states

            :attr:`a_ref` (:attr:`float`): if :attr:`ref_state` is not a current state,
            must be provided. This is the new reference's horizontal shift with respect
            to the current reference. Defaults to 1.

            :attr:`b_ref` (:attr:`float`): if :attr:`ref_state` is not a current state,
            must be provided. This is the new reference's vertical shift with respect
            to the current reference. Defaults to 1.
        """
        if len(self.htransforms) == 1:
            if self.htransforms[0].type == "Multiply":
                if ref_state in self.states and a_ref == 1:
                    # Uncertainty propagation
                    ind_ref = self.states.index(ref_state)
                    if self.huncertainties[0][0] != 0:
                        print("Warning: reference state has already been changed. Uncertainty estimates may no longer be accurate")
                    else:
                        huncertainties_pairwise = [0]
                        for i in range(1, len(self.huncertainties[0])):
                            huncertainties_pairwise += [self.hparams[0][i]
                                    *np.sqrt((self.huncertainties[0][i]/self.hparams[0][i])**2
                                        - (self.huncertainties[0][i-1]/self.hparams[0][i-1])**2)]
                        self.huncertainties[0][ind_ref] = 0
                        for i in range(ind_ref-1, -1, -1):
                            self.huncertainties[0][i] = (self.hparams[0][i]
                                    *np.sqrt((huncertainties_pairwise[i]/self.hparams[0][i])**2
                                        + (self.huncertainties[0][i+1]/self.hparams[0][i+1])**2))
                        for i in range(ind_ref+1, len(self.huncertainties[0])):
                            self.huncertainties[0][i] = (self.hparams[0][i]
                                    *np.sqrt((huncertainties_pairwise[i]/self.hparams[0][i])**2
                                        + (self.huncertainties[0][i-1]/self.hparams[0][i-1])**2))

                    # Find the reference shift factor
                    a_ref = self.hparams[0][ind_ref]

                # Rescale the shift factors and uncertainties
                for k in range(len(self.hparams[0])):
                    self.hparams[0][k] = self.hparams[0][k]/a_ref
                    self.huncertainties[0][k] = self.huncertainties[0][k]/a_ref

        # Now vertical shifts
        if len(self.vtransforms) == 1:
            if self.vtransforms[0].type == "Multiply":
                if ref_state in self.states and b_ref == 1:
                    # Uncertainty propagation
                    ind_ref = self.states.index(ref_state)
                    if self.vuncertainties[0][0] != 0:
                        print("Warning: reference state has already been changed. Uncertainty estimates may no longer be accurate")
                    else:
                        vuncertainties_pairwise = [0]
                        for i in range(1, len(self.vuncertainties[0])):
                            vuncertainties_pairwise += [self.vparams[0][i]
                                    *np.sqrt((self.vuncertainties[0][i]/self.vparams[0][i])**2
                                        - (self.vuncertainties[0][i-1]/self.vparams[0][i-1])**2)]
                        self.vuncertainties[0][ind_ref] = 0
                        for i in range(ind_ref-1, -1, -1):
                            self.vuncertainties[0][i] = [self.vparams[0][i]
                                    *np.sqrt((vuncertainties_pairwise[i]/self.vparams[0][i])**2
                                        + (self.vuncertainties[0][i+1]/self.vparams[0][i+1])**2)]
                        for i in range(ind_ref+1, len(self.vuncertainties[0])):
                            self.vuncertainties[0][i] = [self.vparams[0][i]
                                    *np.sqrt((vuncertainties_pairwise[i]/self.vparams[0][i])**2
                                        + (self.vuncertainties[0][i-1]/self.vparams[0][i-1])**2)]

                    # Find the reference shift factor
                    b_ref = self.vparams[0][self.states.index(ref_state)]

                # Rescale the shift factors and uncertainties
                for k in range(len(self.vparams[0])):
                    self.vparams[0][k] = self.vparams[0][k]/b_ref
                    self.vuncertainties[0][k] = self.vuncertainties[0][k]/b_ref
        self._shift_data()

    def output_table(self, file=None):
        r"""
        Write a csv file with a table of all parameters (and return as a data frame).

        Args:
            :attr:`file` (:attr:`string`): (optional) path to the file to which
            the data frame will be written

        Returns:
            :attr:`df` (:attr:`pandas.DataFrame`): data frame containing the transformation
            parameters
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
        r"""
        Compute the score of a Monte-Carlo cross validation test.

        Args:
            :attr:`x1`, :attr:`y1`, :attr:`s1` (:attr:`array_like`, :attr:`array_like`,
            :attr:`float`): the independent coordinates, dependent coordinates, and state
            for the first data set

            :attr:`x2`, :attr:`y2`, :attr:`s2` (:attr:`array_like`, :attr:`array_like`,
            :attr:`float`): the independent coordinates, dependent coordinates, and state
            for the second data set

            :attr:`lamh`, :attr:`lamv` (:attr:`list[float]`, :attr:`list[float]`): horizontal
            and vertical hyperparameters for the prior distributions over parameters

            :attr:`alpha` (:attr:`float`): the fraction of points kept in the training set
            :attr:`folds` (:attr:`int`): the number of MCCV folds

        Returns:
            :attr:`score` (:attr:`float`): the MCCV score (averaged over folds)
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
        r"""
        Perform hyperparameter optimization on the prior (regularization) using MCCV.

        Args:
            :attr:`lamh` (:attr:`list[tuple[float]]`): ranges (tuples) for the horizontal
            hyperparameter search. If not searching this parameter, then the entry should be None.

            :attr:`lamv` (:attr:`list[tuple[float]]`): ranges (tuples) for the vertical
            hyperparameter search. If not searching this parameter, then the entry should be None.

            :attr:`npoints` (:attr:`int`): number of grid points to search. Default is 100.

            :attr:`alpha` (:attr:`float`): keep rate for MCCV. Default is 0.5.

            :attr:`folds` (:attr:`int`): number of MCCV folds. Default is 10.

        Returns:
            :attr:`lamh_opt` (:attr:`list[float]`): optimal horizontal hyperparameters
            at each state

            :attr:`lamv_opt` (:attr:`list[float]`): optimal vertical hyperparameters
            at each state
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
