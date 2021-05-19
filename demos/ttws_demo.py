import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
sys.path.append('../src')
from transforms import Multiply
from mastercurve import MasterCurve
from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel, ConstantKernel, DotProduct

# Loop through different RH values for time-temperature superposition
rhs = [70, 80, 85, 90, 95]
h2o = [18.6, 21.6, 23.6, 26.2, 30.4]
ws_tts = []
Gps_tts = []
Gpps_tts = []
tans_tts = []
Ts = [[50, 45, 40, 35, 30, 25, 20], [45, 40, 35, 30, 25, 20], [40, 35, 30, 25, 20],
        [30, 27.5, 25, 22.5, 20],[32.5, 30, 27.5, 25]]
fig1, ax1 = plt.subplots(1,1)
fig2, ax2 = plt.subplots(1,1)
markers = ['s', 'o', '^', 'v', 'D']
colors = ['#000000', '#E93323', '#1200F5', '#398223', '#EF8A34']

for rh in rhs:
    # Read the data
    T = Ts[rhs.index(rh)]
    ws = [[] for i in range(len(T))]
    Gps = [[] for i in range(len(T))]
    Gpps = [[] for i in range(len(T))]
    tans = [[] for i in range(len(T))]

    with open(f"data/ttw/ttw_{rh}.csv") as file:
        reader = csv.reader(file)
        next(reader)
        next(reader)
        for row in reader:
            for k in range(len(T)):
                if not row[2*k+1] == "":
                    ws[k] += [float(row[0])]
                    Gps[k] += [float(row[2*k+1])]
                    Gpps[k] += [float(row[2*k+2])]
    for k in range(len(T)):
        ws[k] = np.log(np.array(ws[k]))
        Gps[k] = np.array(Gps[k])
        Gpps[k] = np.array(Gpps[k])
        tans[k] = Gpps[k]/Gps[k]
        Gps[k] = np.log(Gps[k])
        Gpps[k] = np.log(Gpps[k])

    # Build a master curve
    ws.reverse()
    Gps.reverse()
    Gpps.reverse()
    tans.reverse()
    T.reverse()
    mc = MasterCurve()
    mc.add_data(ws, tans, T)

    # Add transformations
    mc.add_htransform(Multiply())

    # First, do horizontal shifting only
    mc.superpose()
    hparams = np.array(mc.hparams[0])
    hparams_30ref = hparams/hparams[T.index(30)]

    # Get the shifted time coordinates
    wtransformed = mc.xtransformed
    w_tts = wtransformed[0]
    for i in range(1,len(wtransformed)):
        w_tts = np.append(w_tts, wtransformed[i])
    w_tts -= np.log(hparams[T.index(30)])
    ws_tts += [w_tts]

    # Next, do vertical shifting
    mc_vert = MasterCurve()
    mc_vert.add_data(wtransformed, Gps, T)

    # Add transformations
    mc_vert.add_vtransform(Multiply())

    # Now, do vertical shifting only
    mc_vert.superpose()
    vparams = np.array(mc_vert.vparams[0])
    vparams_30ref = vparams/vparams[T.index(30)]

    # Get the shifted Gp, Gpp, and tan
    Gp_tts = Gps[0] + np.log(vparams_30ref[0])
    Gpp_tts = Gpps[0] + np.log(vparams_30ref[0])
    tan_tts = tans[0]
    for i in range(1,len(vparams)):
        Gp_tts = np.append(Gp_tts, Gps[i] + np.log(vparams_30ref[i]))
        Gpp_tts = np.append(Gpp_tts, Gpps[i] + np.log(vparams_30ref[i]))
        tan_tts = np.append(tan_tts, tans[i])
    Gps_tts += [Gp_tts]
    Gpps_tts += [Gpp_tts]
    tans_tts += [tan_tts]

    # Plot the shifts
    ax1.plot(1/(np.array(T) + 273) - 1/303, np.log(hparams_30ref), marker=markers[rhs.index(rh)],
            color=colors[rhs.index(rh)], ls="none", label=str(rh))
    ax2.plot(T, vparams_30ref, marker=markers[rhs.index(rh)],
            color=colors[rhs.index(rh)], ls="none", label=str(rh))

    # Compute Arrhenius activation energy
    x = 1/(np.array(T) + 273)
    y = np.log(hparams_30ref)
    A = np.array([x, np.ones(x.shape)]).T
    res = np.linalg.lstsq(A, y)
    E = res[0][0]*8.314

ax1.set_xlim([-2.5E-4, 2E-4])
ax1.legend()
ax2.set_xlim([17.5, 52.5])
ax2.legend()

# Now, do the water superposition
mc = MasterCurve()
mc.add_data(ws_tts, tans_tts, rhs)

# Add transformations
mc.add_htransform(Multiply(bounds=(1E-4,1)))

# First, do horizontal shifting only
mc.superpose()
hshifts = mc.hparams[0]
hshifts_90ref = hshifts/hshifts[rhs.index(90)]

# Now, do vertical shifting
wtransformed = mc.xtransformed
mc.clear()
mc.add_data(wtransformed, Gps_tts, rhs)
mc.add_vtransform(Multiply())
mc.superpose()

# Plot the shifts
fig3, ax3 = plt.subplots(1,1)
vshifts = mc.vparams[0]
vshifts_90ref = vshifts/vshifts[rhs.index(90)]
ax3.plot(np.array(h2o) - h2o[rhs.index(85)], np.log(hshifts/hshifts[rhs.index(85)]), 'ks')

# Get the shifted coordinates
wtransformed_90 = [w - np.log(hshifts[rhs.index(90)]) for w in wtransformed]
Gptransformed_90 = [Gp - np.log(vshifts[rhs.index(90)]) for Gp in mc.ytransformed]
Gpptransformed_90 = [Gpps_tts[0] + np.log(vshifts_90ref[0])]
for i in range(1,len(vshifts_90ref)):
    Gpptransformed_90 += [Gpps_tts[i] + np.log(vshifts_90ref[i])]

# Plot the master curve
fig4, ax4 = plt.subplots(1,1)
ax5 = ax4.twinx()
for i in range(len(wtransformed)):
    ax5.loglog(np.exp(wtransformed_90[i]), np.exp(Gpptransformed_90[i] - Gptransformed_90[i]),
            marker=markers[i], color=colors[i], ls="none", fillstyle="bottom")
    ax5.loglog(np.exp(wtransformed_90[i]), np.exp(Gpptransformed_90[i] - Gptransformed_90[i]),
            marker=markers[i], mec=colors[i], mfc="w", ls="none", fillstyle="top")
    ax4.loglog(np.exp(wtransformed_90[i]), np.exp(Gptransformed_90[i]), marker=markers[i], color=colors[i],
            ls="none", fillstyle="full", label=rhs[i])
    ax4.loglog(np.exp(wtransformed_90[i]), np.exp(Gpptransformed_90[i]), marker=markers[i], mec=colors[i],
            mfc="w", ls="none")

ax4.legend()
ax4.set_ylim([1,1E4])
ax5.set_ylim([1E-2,1E1])
ax4.set_xlim([1E-7,1E14])
plt.show()
