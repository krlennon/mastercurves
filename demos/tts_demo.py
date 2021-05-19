import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
sys.path.append('../src')
from transforms import Multiply
from mastercurve import MasterCurve

# Read the data
T = [97, 100.6, 101.8, 104.5, 106.7, 109.6, 114.5, 125, 133.8, 144.9]
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
    Js[k] = np.log(np.array(Js[k]))

# Build a master curve
mc = MasterCurve()
mc.add_data(ts, Js, T)

# Add transformations
mc.add_htransform(Multiply())

# Superpose
mc.superpose()

# Plot
fig1, ax1, fig2, ax2, fig3, ax3 = mc.plot()
plt.show()

