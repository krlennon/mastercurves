import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
sys.path.append('../src')
from transforms import Multiply
from mastercurve import MasterCurve
import progressbar

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
mc.plot()
plt.show()

