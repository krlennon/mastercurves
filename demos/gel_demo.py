import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import csv
import sys
from mastercurves import MasterCurve
from mastercurves.transforms import Multiply
from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel
plt.rcParams.update({"font.size": 12})

# Read the data
t = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115]
ts = [[] for i in range(len(t))]
msds = [[] for i in range(len(t))]
with open("data/MSD_MAX1.csv") as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        for k in range(len(t)):
            if not row[2*k] == "":
                ts[k] += [float(row[2*k])]
                msds[k] += [float(row[2*k+1])]
for k in range(len(t)):
    ts[k] = np.log(np.array(ts[k]))
    msds[k] = np.log(np.array(msds[k]))

## Build a master curve in the forward direction
#mc = MasterCurve()
#mc.add_data(ts, msds, t)
#mc.set_gp_kernel(RationalQuadratic() + WhiteKernel())
#
## Add transformations
#mc.add_htransform(Multiply(bounds=(1E-2,1), prior="Gaussian"))
#mc.add_vtransform(Multiply())
#
## Run hyperparameter optimization
##lamh_f, lamv_f = mc.hpopt(lamh=(0.01,10.0), npoints=16, alpha=0.1, folds=20)
#
## Superpose
##lamh_f = [6.309573444801936, 10.0, 3.981071705534973, 10.0, 6.309573444801936, 1.584893192461114, 3.981071705534973, 10.0, 0.15848931924611143, 0.06309573444801933, 0.01, 0.6309573444801934, 0.25118864315095807, 0.3981071705534973, 6.309573444801936, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
lamh_f = [6.309573444801936, 10.0, 0, 10.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
lamv_f = list(np.zeros(len(lamh_f)))
#losses_fwd = mc.superpose(lamh_f, lamv_f)
##losses_fwd = mc.superpose()
#hparams_fwd = mc.hparams[0]
#vparams_fwd = mc.vparams[0]
#
## Build a master curve in the backwards direction
#mc.clear()
#mc.set_gp_kernel(RationalQuadratic() + WhiteKernel())
#ts.reverse()
#msds.reverse()
#t.reverse()
#mc.add_data(ts, msds, t)
#
## Add transformations
#mc.add_htransform(Multiply(bounds=(1E-2,1), prior="Gaussian"))
#mc.add_vtransform(Multiply())
#
## Run hyperparameter optimization
##lamh_b, lamv_b = mc.hpopt(lamh=(0.01,10.0), npoints=16, alpha=0.1, folds=20)
##print(lamh_f)
##print(lamh_b)
#
## Superpose
##lamh_b = [0.01, 10.0, 0.1, 0.015848931924611134, 0.25118864315095807, 0.1, 6.309573444801936, 10.0, 10.0, 6.309573444801936, 10.0, 3.981071705534973, 6.309573444801936, 6.309573444801936, 6.309573444801936, 3.981071705534973, 3.981071705534973, 0.25118864315095807, 10.0, 10.0, 10.0]
##lamv_b = list(np.zeros(len(lamh_b)))
##losses_bwd = mc.superpose(lamh_b, lamv_b)
#losses_bwd = mc.superpose()
#hparams_bwd = mc.hparams[0]
#vparams_bwd = mc.vparams[0]
#
## Find the lowest total loss for splitting to two master curves
#losses_total = []
#for i in range(len(losses_fwd)):
#    losses_total += [sum(losses_fwd[:i]) + sum(losses_bwd[:-1-i])]
#ind_split = np.argmin(np.array(losses_total))
#
## Plot
#fig, ax = plt.subplots(1,1)
#t.reverse()
#ax.semilogy(t[:ind_split+1], hparams_fwd[:ind_split+1], 'ro-')
#ax.semilogy(t[:ind_split+1], vparams_fwd[:ind_split+1], 'b^-')
#ax.semilogy(t[-1:ind_split-len(t):-1], hparams_bwd[:len(t)-ind_split-1], 'ro-', fillstyle="none")
#ax.semilogy(t[-1:ind_split-len(t):-1], vparams_bwd[:len(t)-ind_split-1], 'b^-', fillstyle="none")
#ax.set_ylim([2E-7,9])
#ax.tick_params(which="both", direction="in", top=True, right=True)
#fig.savefig("gel_shifts.pdf", transparent=True)

# Construct and plot the separate master curves
mc = MasterCurve()
ind_split = 14
mc.add_data(ts[:ind_split+1], msds[:ind_split+1], t[:ind_split+1])
mc.set_gp_kernel(RationalQuadratic() + WhiteKernel())

# Add transformations
mc.add_htransform(Multiply(bounds=(1E-2,1), prior="Gaussian"))
mc.add_vtransform(Multiply())

# Run hyperparameter optimization
#lamh_f, lamv_f = mc.hpopt(lamh=(0.01,10.0), npoints=16, alpha=0.1, folds=20)

# Superpose
losses_fwd = mc.superpose(lamh_f, lamv_f)
fig1, ax1, fig2, ax2, fig3, ax3 = mc.plot(log=True, colormap=plt.cm.viridis)
ax3.tick_params(which="both",direction="in",top=True,right=True)
ax3.xaxis.set_minor_locator(tck.LogLocator(base=10, subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0), numticks=10))
ax3.set_xlim([2E-8,20])
ax3.set_ylim([1E-6,20])
fig3.savefig("gel_mc_pre.pdf", transparent=True)

plt.show()

