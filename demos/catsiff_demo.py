import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import csv
import sys
from mastercurves import MasterCurve
from mastercurves.transforms import Multiply
import time

# Read the data
T = [-82.6, -81.7, -80.8, -79.3, -77.3, -76.7, -76.5, -74.1, -71.1, -70.6, -69.15, -66.5, -65.4, -62.0, -60.9, -58.8, -56.2, -54.0, -51.4, -49.6, -44.3, -40.1, -20.2]
T.reverse()
ts = [[] for i in range(len(T))]
Es_uncorrected = [[] for i in range(len(T))]
Es = [[] for i in range(len(T))]
log10Eg_ref = 10.48
T_ref = 298
with open("data/catsiff_tts.csv") as file:
    reader = csv.reader(file)
    next(reader)
    next(reader)
    for row in reader:
        for k in range(len(T)):
            if not row[2*k] == "":
                ts[k] += [float(row[2*k])]
                Es[k] += [float(row[2*k+1])]
for k in range(len(T)):
    ts[k] = np.log(np.array(ts[k]))
    logE_uncorrected = np.log(np.array(Es[k]))
    Es_uncorrected[k] = logE_uncorrected
    logW = logE_uncorrected - np.log(10**log10Eg_ref) + np.log(1 - (T[k] + 273)/T_ref)
    Es[k] = logE_uncorrected - np.log((T[k] + 273)/T_ref + np.exp(logW))
ts.reverse()
Es.reverse()
Es_uncorrected.reverse()

# Build a master curve
mc = MasterCurve()
mc.add_data(ts, Es, T)
mc.set_gp_kernel(mc.kernel)

# Add transformations
mc.add_htransform(Multiply())

# Superpose
start = time.time()
mc.superpose()
stop = time.time()
print(f"Time: {stop-start} s")

# Plot
mc.change_ref(25, a_ref=mc.hparams[0][-1]*10**(12.05))
colormap = plt.cm.viridis_r
fig1, ax1 = plt.subplots(1,1)
for k in range(len(mc.xtransformed)):
    ax1.loglog(np.exp(mc.xtransformed[k])*3600, np.exp(mc.ytransformed[k])*0.1, 'o', label=str(mc.states[k]),
            color=colormap(k/len(mc.states)))
ax1.tick_params(which="both",direction="in",right=True,top=True)
ax1.xaxis.set_major_locator(tck.LogLocator(base=10.0, numticks=14))
ax1.xaxis.set_minor_locator(tck.LogLocator(base=10.0, subs=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],numticks=14))
ax1.set_xlabel(r"$t$ $(s)$")
ax1.set_ylabel(r"$E_{r,T}(t)$ $(Pa)$")
#ax1.set_xticklabels(["", "", "", r"$10^{-12}$", "", r"$10^{-10}$", "", r"$10^{-8}$", "", r"$10^{-6}$", "", r"$10^{-4}$", "", r"$10^{-2}$"])

# Compare to the WLF form from Plazek
a = mc.hparams[0]
log10a_catsiff = [2.34, 4.20, 4.72, 5.31, 5.47, 5.77, 6.04, 6.55, 6.79, 7.06, 7.62, 7.93, 8.46, 8.85, 9.13, 9.70, 10.47, 10.70, 10.89, 11.51, 11.53, 11.89, 12.05]
T_array = np.array(T)
#a_mod = 10**(-np.log10(a) + np.log10(a[-1]) + 1.13)
da = np.array(mc.huncertainties[0])
da_mod = da/(np.array(a)**2)
fig2, ax2 = plt.subplots(1,1)
#ax2.semilogy(1/(T_array[1:] + 273), 10**np.array(log10a_catsiff[1:]), 'ko', fillstyle="none")
ax2.semilogy(1/(T_array[1:] + 273), 1/np.array(a[1:]), 'ko')
ax2.tick_params(which="both",direction="in",top=True,right=True)
ax2.yaxis.set_major_locator(tck.LogLocator(base=10.0, numticks=12))
ax2.yaxis.set_minor_locator(tck.LogLocator(base=10.0, subs=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], numticks=12))
ax2.set_xlabel(r"$1/T$ $(K^{-1})$")
ax2.set_ylabel(r"$a_{298}$")
#ax2.set_yticklabels(["", r"$10^{-8}$", "", r"$10^{-6}$", "", r"$10^{-4}$", "", r"$10^{-2}$", "", r"$10^{0}$", "", r"$10^{2}$"])

# Make GPs for the raw data for plotting
mc2 = MasterCurve()
mc2.add_data(ts, Es_uncorrected, T)
mc2.set_gp_kernel(mc2.kernel)
mc2._fit_gps()
fig3, ax3 = plt.subplots(1,1)
for k in range(len(mc2.xdata)):
    xgp = np.linspace(np.min(mc2.xdata[k]),np.max(mc2.xdata[k]),100)
    y = mc2.gps[k].predict(xgp.reshape(-1,1), return_std=False)
    ax3.loglog(np.exp(xgp)*3600, np.exp(y)*0.1, color=colormap(k/len(mc2.states)))
    ax3.loglog(np.exp(mc2.xdata[k])*3600, np.exp(mc2.ydata[k])*0.1, 'o', label=str(mc2.states[k]),
            color=colormap(k/len(mc2.states)))
ax3.tick_params(which="both",direction="in",right=True,top=True)
ax3.set_xlabel(r"$t/a_{298}$ $(s)$")
ax3.set_ylabel(r"$E_{r,298}(t)$ $(Pa)$")

plt.show()
