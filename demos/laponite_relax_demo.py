import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
from mastercurves import MasterCurve
from mastercurves.transforms import Multiply, PowerLawAge
import matplotlib.ticker as tck
import time
from sklearn.gaussian_process.kernels import WhiteKernel

# Read the files
ts = [[],[],[],[],[]]
Js = [[],[],[],[],[]]
with open("data/joshi_modulus_25.csv") as file:
    reader = csv.reader(file)
    next(reader)
    next(reader)
    next(reader)
    next(reader)
    next(reader)
    next(reader)
    for row in reader:
        for k in range(5):
            if not row[2*k] == "":
                ts[k] += [float(row[2*k])]
                Js[k] += [float(row[2*k + 1])]

for k in range(5):
    ts[k] = np.log(np.array(ts[k]))
    Js[k] = np.log(np.array(Js[k]))

tws = [600, 1200, 1800, 2400, 3600]

# Develop a master curve
mc = MasterCurve()
mc.add_data(ts, Js, tws)

# Add transformations
mc.add_htransform(PowerLawAge(600))
mc.add_vtransform(Multiply())

# Superpose
start = time.time()
mc.superpose()
stop = time.time()
print(f"Time: {stop-start}s")
print(mc.hparams[0])
print(mc.huncertainties[0])

# Plot
fig1, ax1, fig2, ax2, fig3, ax3 = mc.plot(log=True, colormap=plt.cm.viridis)
ax2.tick_params(which="both",direction="in",top=True,right=True)
ax2.yaxis.set_major_locator(tck.MultipleLocator(100))
ax2.yaxis.set_minor_locator(tck.MultipleLocator(10))
ax2.yaxis.set_minor_formatter(tck.FormatStrFormatter(''))
ax3.tick_params(which="both",direction="in",top=True,right=True)
ax3.yaxis.set_major_locator(tck.MultipleLocator(100))
ax3.yaxis.set_minor_locator(tck.MultipleLocator(10))
ax3.yaxis.set_minor_formatter(tck.FormatStrFormatter(''))
plt.show()

