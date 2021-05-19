import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
sys.path.append('../src')
from transforms import Multiply, PowerLawAge
from mastercurve import MasterCurve

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
mc.add_htransform(PowerLawAge(15000))
mc.add_vtransform(Multiply())

# Superpose
mc.superpose()

# Plot
fig1, ax1, fig2, ax2, fig3, ax3 = mc.plot()
plt.show()

