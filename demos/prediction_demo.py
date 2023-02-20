import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
from mastercurves import MasterCurve
from mastercurves.transforms import Multiply
import progressbar
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel, ConstantKernel
plt.rcParams.update({"font.size": 12})

# Read the data
phi = [0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80]
gdots = [[] for i in range(len(phi))]
sigmas = [[] for i in range(len(phi))]
with open("data/emulsions_v2.csv") as file:
    reader = csv.reader(file)
    next(reader)
    next(reader)
    for row in reader:
        for k in range(len(phi)):
            if not row[2*k] == "":
                gdots[k] += [float(row[2*k])]
                sigmas[k] += [float(row[2*k+1])]

# Remove the phi = 0.72 data set
phi.remove(0.72)
gdot_072 = gdots[2]
sigma_072 = sigmas[2]
gdots.remove(gdot_072)
sigmas.remove(sigma_072)

# Hyperparameter optimization on the viscosity
etav = np.linspace(0,0.1,100)
losses = []

# Define a master curve
mc = MasterCurve()

for it in progressbar.progressbar(range(len(etav))):
    eta = etav[it]
    gdots_new = [[] for i in range(len(phi))]
    sigmas_new = [[] for i in range(len(phi))]
    for k in range(len(phi)):
        gdots_new[k] = np.array(gdots[k])
        sigmas_new[k] = np.array(sigmas[k]) - eta*gdots_new[k]
        gdots_new[k] = np.log(gdots_new[k])
        sigmas_new[k] = np.log(sigmas_new[k])

    # Build a master curve
    mc.clear()
    mc.add_data(gdots_new, sigmas_new, phi)

    # Add transformations
    mc.add_htransform(Multiply())
    mc.add_vtransform(Multiply())

    # Superpose
    loss = mc.superpose()
    losses += [np.sum(np.array(loss))]

# Create a master curve at the optimal eta
ind_min = np.argmin(losses)
eta_min = etav[ind_min]
print(eta_min)

# Compute the uncertainty in eta from the inverse Hessian
losses = np.array(losses)
deta = etav[1] - etav[0]
hess = (losses[2:] - 2*losses[1:-1] + losses[:-2])/(deta**2)
eta_uncertainty = np.sqrt(1/hess[ind_min-1])
print(eta_uncertainty)

gdots_new = [[] for i in range(len(phi))]
sigmas_new = [[] for i in range(len(phi))]
for k in range(len(phi)):
    gdots_new[k] = np.array(gdots[k])
    sigmas_new[k] = np.array(sigmas[k]) - eta_min*gdots_new[k]
    gdots_new[k] = np.log(gdots_new[k])
    sigmas_new[k] = np.log(sigmas_new[k])

# Build a master curve
mc.clear()
mc.add_data(gdots_new, sigmas_new, phi)

# Add transformations
mc.add_htransform(Multiply())
mc.add_vtransform(Multiply())

# Superpose
mc.superpose()
datafig, dataax, gpfig, gpax, mcfig, mcax = mc.plot(log=True, colormap=plt.cm.viridis)
gpax.tick_params(which="both", direction="in", right=True, top=True)
mcax.tick_params(which="both", direction="in", right=True, top=True)

# Get the inferred parameters and uncertainties
b = mc.vparams[0]
db = mc.vuncertainties[0]
a = mc.hparams[0]
da = mc.huncertainties[0]
sigma_y = 4.7/np.array(b)
dsigma_y = 4.7*np.array(db)/(np.array(b)**2)
gamma_c = 4.7/np.array(a)
dgamma_c = 4.7*np.array(da)/(np.array(a)**2)
fig, ax = plt.subplots(1,1)
ax.errorbar(phi, sigma_y, yerr=dsigma_y, color="blue", linestyle="none", marker="^")
ax.errorbar(phi, gamma_c, yerr=dgamma_c, color="red", linestyle="none", marker="o")
ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)

# Fit the shift factors to GP models
kernel = RationalQuadratic() * ConstantKernel() + ConstantKernel() + WhiteKernel()
gp_sigma_y = GaussianProcessRegressor(kernel=kernel +
        WhiteKernel(np.max(dsigma_y)**2, "fixed")).fit(np.array(phi).reshape(-1,1), sigma_y)
gp_gamma_c = GaussianProcessRegressor(kernel=kernel +
        WhiteKernel(np.max(dgamma_c)**2, "fixed")).fit(np.array(phi).reshape(-1,1), gamma_c)

# Predict the yield stress and critical shear rate at phi = 0.72
s, ds = gp_sigma_y.predict(np.array(0.72).reshape(-1,1), return_std=True)
print(s)
print(ds)
g, dg = gp_gamma_c.predict(np.array(0.72).reshape(-1,1), return_std=True)
print(g)
print(dg)

# Add to plot
phi_lim = ax.get_xlim()
phi_v = np.linspace(phi_lim[0], phi_lim[1])
s_v, ds_v = gp_sigma_y.predict(phi_v.reshape(-1,1), return_std=True)
g_v, dg_v = gp_gamma_c.predict(phi_v.reshape(-1,1), return_std=True)
ax.plot(phi_v, s_v, 'b')
ax.plot(phi_v, g_v, 'r')
ax.fill_between(phi_v, s_v - ds_v, s_v + ds_v, color="b", alpha=0.2)
ax.fill_between(phi_v, g_v - dg_v, g_v + dg_v, color="r", alpha=0.2)
ax.set_xlim(phi_lim)

# Plot the master curve after shifting the reference to correspond with
# the yield stress and critical shear rate at phi = 0.68
mc.change_ref(0.68, 4.7, 4.7)

# Get the transformed data (in log space)
gdot_transformed = mc.xtransformed
gdot_transformed = np.concatenate(gdot_transformed)
sigma_transformed = mc.ytransformed
sigma_transformed = np.concatenate(sigma_transformed)

# Fit a new GP for the master curve
gp_master = GaussianProcessRegressor(kernel=kernel).fit(gdot_transformed.reshape(-1,1), sigma_transformed)
gdot_transformed_v = np.linspace(np.min(gdot_transformed) - 1, np.max(gdot_transformed) + 1, 100)
sigma_transformed_v, dsigma_transformed_v = gp_master.predict(gdot_transformed_v.reshape(-1,1), return_std=True)

# Transform using the predicted shift factors
gdot_v = np.exp(gdot_transformed_v)*g
sigma_v = np.exp(sigma_transformed_v)*s
dsigma_v = np.sqrt((s*dsigma_transformed_v)**2 + (np.exp(sigma_transformed_v)*ds)**2)
dsigma_v = np.sqrt((dsigma_v)**2 + (eta_uncertainty*gdot_v)**2)
dgdx = np.zeros(np.shape(sigma_transformed_v))
dgdx[1:-1] = (sigma_transformed_v[2:] - sigma_transformed_v[:-2])/(gdot_transformed_v[2:] - gdot_transformed_v[:-2])
dgdx[0] = (sigma_transformed_v[1] - sigma_transformed_v[0])/(gdot_transformed_v[1] - gdot_transformed_v[0])
dgdx[-1] = (sigma_transformed_v[-1] - sigma_transformed_v[-2])/(gdot_transformed_v[-1] - gdot_transformed_v[-2])
dsigma_v = np.sqrt((dsigma_v)**2 + (s*dgdx*gdot_v*dg/(g**2)))

# Plot the predictions
fig2, ax2 = plt.subplots(1,1)
ax2.loglog(gdot_v, sigma_v + eta_min*gdot_v, color=plt.cm.viridis(2/7))
ax2.fill_between(gdot_v, sigma_v + eta_min*gdot_v - dsigma_v, sigma_v + eta_min*gdot_v + dsigma_v,
        color=plt.cm.viridis(2/7), alpha=0.3)
ax2.loglog(gdot_072, sigma_072, 'o', color=plt.cm.viridis(2/7))
ax2.set_xlim([5E-4, 2E3])
ax2.tick_params(which="both",direction="in",top=True,right=True)

plt.show()

