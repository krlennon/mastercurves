import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
sys.path.append('../src')
from transforms import Multiply
from mastercurve import MasterCurve
from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel

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

# Build a master curve in the forward direction
mc = MasterCurve()
mc.add_data(ts, msds, t)
mc.set_gp_kernel(RationalQuadratic() + WhiteKernel())

# Add transformations
mc.add_htransform(Multiply())
mc.add_vtransform(Multiply())

# Superpose
losses_fwd = mc.superpose()
hparams_fwd = mc.hparams[0]
vparams_fwd = mc.vparams[0]

# Build a master curve in the backwards direction
mc.clear()
mc.set_gp_kernel(RationalQuadratic() + WhiteKernel())
ts.reverse()
msds.reverse()
t.reverse()
mc.add_data(ts, msds, t)

# Add transformations
mc.add_htransform(Multiply())
mc.add_vtransform(Multiply())

# Superpose
losses_bwd = mc.superpose()
hparams_bwd = mc.hparams[0]
vparams_bwd = mc.vparams[0]

# Find the lowest total loss for splitting to two master curves
losses_total = []
for i in range(len(losses_fwd)):
    losses_total += [sum(losses_fwd[:i]) + sum(losses_bwd[:-1-i])]
ind_split = np.argmin(np.array(losses_total))

# Plot
fig, ax = plt.subplots(1,1)
t.reverse()
ax.semilogy(t[:ind_split+1], hparams_fwd[:ind_split+1], 'ro-')
ax.semilogy(t[:ind_split+1], vparams_fwd[:ind_split+1], 'b^-')
ax.semilogy(t[-1:ind_split-len(t):-1], hparams_bwd[:len(t)-ind_split-1], 'ro-', fillstyle="none")
ax.semilogy(t[-1:ind_split-len(t):-1], vparams_bwd[:len(t)-ind_split-1], 'b^-', fillstyle="none")
plt.show()

