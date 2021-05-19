import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel, ConstantKernel
from scipy import optimize
import pandas as pd

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
        self.kernel = RationalQuadratic() * ConstantKernel() + ConstantKernel() + WhiteKernel()
        self.gps = []

        # Set the default x and y transformations to the identity
        self.htransforms = []
        self.hparam_names = []
        self.hparams = []
        self.hbounds = []
        self.hshared = []
        self.vtransforms = []
        self.vparam_names = []
        self.vparams = []
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
        self.hbounds = []
        self.hshared = []
        self.vtransforms = []
        self.vparam_names = []
        self.vparams = []
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

    def _pairwise_nlog_posterior(self, ind1, ind2, hp1, hp2, vp1, vp2, hprior=None, vprior=None):
        """
        Compute the pairwise negative log likelihood loss between two curves.
        Inputs:
            ind1, ind2 - the indices in the data lists corresponding to the two data sets.
            hp1, hp2 - the horizontal shift parameters for each data set
            vp1, vp2 - the vertical shift parameters for each data set
            hprior, vprior - list of functions defining the negative log of the prior over the params
        Outputs:
            loss - the negative log of the posterior, computed over these two data sets.
        """
        # Get the data and GPs
        x1, y1, s1, gp1 = self._unpack(ind1)
        x2, y2, s2, gp2 = self._unpack(ind2)

        # Transform to the other state
        x1s2 = self._change_states(self.htransforms, x1, s1, s2, hp1, hp2)

#        # Get bounds for resampling
#        xmin = np.max([np.min(x2), np.min(x1s2)])
#        xmax = np.min([np.max(x2), np.max(x1s2)])
#        if xmin >= xmax:
#            return np.inf
#        x2 = np.linspace(xmin, xmax, 100)
#        x1 = self._change_states(self.htransforms, x2, s2, s1, hp2, hp1)
#        y1 = gp1.predict(x1.reshape(-1,1))
#        y2 = gp2.predict(x2.reshape(-1,1))
#        x2s1 = x1
#        x1s2 = x2

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

        # Compute the posterior loss with Gaussian prior (i.e. L2 regularization)
        loss = np.sum(nll1) + np.sum(nll2) + 0*(len(nll1) + len(nll2))*(np.log(hp1[0]) - np.log(hp2[0]))**2
        return loss

    def _objective(self, inds, hps, vps):
        """
        Compute the loss over all data sets in inds, given params hps and vps.
        Inputs:
            inds - indices of data curves to shift
            hps - horizontal shifting parameters
            vps - vertical shifting parameters
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
            loss += self._pairwise_nlog_posterior(ind1, ind2, hp1, hp2, vp1, vp2)
        return loss

    def superpose(self):
        """
        Optimize the transformations to superpose the data sets onto a single master curve.
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
            else:
                hparams += [hparams[-1] + n - 1]
                bounds += (n - 1)*[htransform.bounds]
                hpval = self.hparams[k]
                self.hparams[k] = nsets*[hpval]
            k += 1
        vparams = [hparams[-1]]
        k = 0
        for vtransform in self.vtransforms:
            if vtransform.type == "Multiply" and len(self.vtransforms) == 1:
                if vtransform.scale == "log":
                    vpval = self.vparams[k]
                    self.vparams[k] = nsets*[vpval]
                    continue
            if vtransform.shared:
                vparams += [vparams[-1] + 1]
                bounds += [vtransform.bounds]
                vpval = self.vparams[k]
                self.vparams[k] = [vpval]
            else:
                vparams += [vparams[-1] + n - 1]
                bounds += (n - 1)*[vtransform.bounds]
                vpval = self.vparams[k]
                self.vparams[k] = nsets*[vpval]
            k += 1

        # Set up the objective for simultaneous shifting of all data sets
        if not pairwise:
            inds = [i for i in range(nsets)]
            obj = lambda p: self._objective(inds, [p[hparams[i-1]:hparams[i]] for i in range(1,len(hparams))],
                    [p[vparams[i-1]:vparams[i]] for i in range(1,len(vparams))])

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

                # Assign the optimal parameter
                for k in range(len(hparams[1:])):
                    if hparams[k+1] - hparams[k] > 0:
                        self.hparams[k] = [p_min]
                for k in range(len(vparams[1:])):
                    if vparams[k+1] - vparams[k] > 0:
                        self.vparams[k] = [p_min]

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
            for m in range(nsets-1):
                inds = [m, m+1]
                obj = lambda p: self._objective(inds, [p[hparams[i-1]:hparams[i]] for i in range(1,len(hparams))],
                        [p[vparams[i-1]:vparams[i]] for i in range(1,len(vparams))])

                # If there's only one parameter to optimize over, then do a linesearch
                if vparams[-1] == 1:
                    # Shift so that ind1 is the new reference state
                    #pv = np.linspace(bounds[0][0], bounds[0][1], 100)
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

                    # Assign the optimal parameter
                    for k in range(len(hparams[1:])):
                        if hparams[k+1] - hparams[k] > 0:
                            self.hparams[k][m+1] = p_min
                    for k in range(len(vparams[1:])):
                        if vparams[k+1] - vparams[k] > 0:
                            self.vparams[k][m+1] = p_min

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

    def plot(self):
        """
        Plot the data, GPs, and master curve.
        """
        # Plot the data
        fig1, ax1 = plt.subplots(1,1)
        for k in range(len(self.xdata)):
            ax1.plot(self.xdata[k], self.ydata[k], 'o', label=str(self.states[k]))
        ax1.legend()

        # Plot the data with the GPs
        fig2, ax2 = plt.subplots(1,1)
        for k in range(len(self.xdata)):
            ax2.plot(self.xdata[k], self.ydata[k], 'o', label=str(self.states[k]))
        z = ax2.legend()
        tm = z.legendHandles
        xlim = ax2.get_xlim()
        xgp = np.linspace(xlim[0],xlim[1],100)
        for k in range(len(self.gps)):
            y, s = self.gps[k].predict(xgp.reshape(-1,1), return_std=True)
            ax2.plot(xgp, y, color=tm[k].get_color())
            ax2.fill_between(xgp, y - s, y + s, color=tm[k].get_color(), alpha=0.2)

        # Plot the master curve
        fig3, ax3 = plt.subplots(1,1)
        for k in range(len(self.xtransformed)):
            ax3.plot(self.xtransformed[k], self.ytransformed[k], 'o', label=str(self.states[k]))
        ax3.legend()

        return fig1, ax1, fig2, ax2, fig3, ax3

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
