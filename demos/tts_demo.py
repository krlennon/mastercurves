import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import csv
import sys
from mastercurves import MasterCurve
from mastercurves.transforms import Multiply
from sklearn.gaussian_process.kernels import WhiteKernel
import time

# Relative noise level
noise = 0.2
np.random.seed(3)

# Read the data
T = [97, 100.6, 101.8, 104.5, 106.7, 109.6, 114.5, 125, 133.8, 144.9]
T.reverse()
ts = [[] for i in range(len(T))]
Js = [[] for i in range(len(T))]
with open("data/tts_plazek.csv") as file:
    reader = csv.reader(file)
    next(reader)
    next(reader)
    for row in reader:
        for k in range(len(T)):
            if not row[2*k] == "":
                ts[k] += [float(row[2*k])]
                Js[k] += [float(row[2*k+1])]
for k in range(len(T)):
    ts[k] = np.log(np.array(ts[k]))
    Js[k] = np.log(np.array(Js[k]) + noise*np.random.randn(len(Js[k]))*np.array(Js[k]))

# Build a master curve
mc = MasterCurve()
mc.add_data(ts, Js, T)
mc.set_gp_kernel(mc.kernel)

# Add transformations
mc.add_htransform(Multiply())

# Superpose
start = time.time()
mc.superpose()
stop = time.time()
print(f"Time: {stop-start} s")

# Compare to the WLF form from Plazek
# Change the reference state to be consistent
mc.change_ref(97)
mc.change_ref(100, a_ref=10**(1.13))

# Get the parameters and uncertainties
a = 1/np.array(mc.hparams[0])
da = np.array(mc.huncertainties[0])*a**2
T = np.array(T)

# Plot
fig, ax = plt.subplots(1,1)
ax.errorbar(T, a, yerr=da, ls='none', marker='o', mfc='k', mec='k')

# Compute WLF predictions and plot
Tv = np.linspace(90,150)
ax.semilogy(Tv, 10**(-10.7*(Tv - 100)/(29.9 + Tv - 100)), 'b')
ax.set_xlim([90,150])
ax.set_ylim([1E-8,1E2])
ax.tick_params(which="both",direction="in",top=True,right=True)
ax.xaxis.set_minor_locator(tck.LinearLocator(numticks=61))
ax.yaxis.set_major_locator(tck.LogLocator(base=10.0, numticks=12))
ax.yaxis.set_minor_locator(tck.LogLocator(base=10.0, subs=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], numticks=12))
ax.set_yticklabels(["", r"$10^{-8}$", "", r"$10^{-6}$", "", r"$10^{-4}$", "", r"$10^{-2}$", "", r"$10^{0}$", "", r"$10^{2}$"])

# Plot the master curve
fig1, ax1, fig2, ax2, fig3, ax3 = mc.plot(log=True, colormap=plt.cm.viridis_r)
ax2.tick_params(which="both",direction="in",right=True,top=True)
ax3.tick_params(which="both",direction="in",right=True,top=True)
ax3.xaxis.set_major_locator(tck.LogLocator(base=10.0, numticks=14))
ax3.xaxis.set_minor_locator(tck.LogLocator(base=10.0, subs=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],numticks=14))
ax3.set_xticklabels(["", "", "", r"$10^{0}$", "", r"$10^{2}$", "", r"$10^{4}$", "", r"$10^{6}$", "", r"$10^{8}$", "", r"$10^{10}$"])

plt.show()
