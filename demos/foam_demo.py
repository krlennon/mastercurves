import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
from mastercurves import MasterCurve
from mastercurves.transforms import Multiply
import progressbar
from sklearn.gaussian_process.kernels import WhiteKernel

# Read the data
t = [100, 200, 300, 400, 500, 600, 700, 800]
phi = [0.85, 0.9, 0.97, 0.99]
vs = [[] for i in range(len(t))]
torques = [[] for i in range(len(t))]
with open("data/HB_model_fit_R01.csv", encoding='mac_roman') as file:
    reader = csv.reader(file)
    next(reader)
    next(reader)
    k = 0
    for row in reader:
        vs[int(np.floor(k/10))] += [float(row[10])]
        torques[int(np.floor(k/10))] += [float(row[11])]
        k += 1

for i in range(len(t)):
    vs[i] = np.log(np.array(vs[i]))
    torques[i] = np.log(np.array(torques[i]))

# Hyperparameter optimization on the viscosity
vs.reverse()
torques.reverse()
t.reverse()

# Define a master curve
mc = MasterCurve()
mc.set_gp_kernel(mc.kernel + WhiteKernel(0.05**2, "fixed"))
mc.add_data(vs, torques, t)

# Add transformations
mc.add_htransform(Multiply())
mc.add_vtransform(Multiply())
mc.superpose()
mc.plot(log=True, colormap=plt.cm.viridis)

# Plot the shifts
hshifts = mc.hparams[0]
hshifts = np.array(hshifts)
hshifts = 1/hshifts
vshifts = mc.vparams[0]
vshifts = np.array(vshifts)
vshifts = 1/vshifts
fig1, ax1 = plt.subplots(1,1)
ax1.plot(t, hshifts, 'bo', fillstyle="none")
fig2, ax2 = plt.subplots(1,1)
ax2.plot(t, vshifts, 'bo', fillstyle="none")

plt.show()
