import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
from mastercurves import MasterCurve
from mastercurves.transforms import Multiply
import progressbar
import multiprocessing

def fit_with_eta(data):
    eta = data[0]
    phi = data[1]
    gdots = data[2]
    sigmas = data[3]
    gdots_new = [[] for i in range(len(phi))]
    sigmas_new = [[] for i in range(len(phi))]
    for k in range(len(phi)):
        gdots_new[k] = np.array(gdots[k])
        sigmas_new[k] = np.array(sigmas[k]) - eta*gdots_new[k]
        gdots_new[k] = np.log(gdots_new[k])
        sigmas_new[k] = np.log(sigmas_new[k])

    # Define a master curve
    mc = MasterCurve()

    # Build a master curve
    mc.clear()
    mc.add_data(gdots_new, sigmas_new, phi)

    # Add transformations
    mc.add_htransform(Multiply())
    mc.add_vtransform(Multiply())

    # Superpose
    loss = np.sum(np.array(mc.superpose()))
    return loss

if __name__ == "__main__":
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
    inputs = [(eta, phi, gdots, sigmas) for eta in list(etav)]
    losses = []

    # Define function for multiprocessess evaluation
    pool = multiprocessing.Pool(10)
    losses = pool.map(fit_with_eta, inputs)

    # Create a master curve at the optimal eta
    ind_min = np.argmin(losses)
    eta_min = etav[ind_min]
    print(eta_min)

    # Compute the uncertainty in eta from the inverse Hessian
    losses = np.array(losses)
    deta = etav[1] - etav[0]
    hess = (losses[2:] - 2*losses[1:-1] + losses[:-2])/(deta**2)
    eta_uncertainty = np.sqrt(1/hess[ind_min-1])
    print(eta_uncertainty)

    gdots_new = [[] for i in range(len(phi))]
    sigmas_new = [[] for i in range(len(phi))]
    for k in range(len(phi)):
        gdots_new[k] = np.array(gdots[k])
        sigmas_new[k] = np.array(sigmas[k]) - eta_min*gdots_new[k]
        gdots_new[k] = np.log(gdots_new[k])
        sigmas_new[k] = np.log(sigmas_new[k])

    # Build a master curve
    mc = MasterCurve()
    mc.add_data(gdots_new, sigmas_new, phi)

    # Add transformations
    mc.add_htransform(Multiply())
    mc.add_vtransform(Multiply())

    # Superpose
    mc.superpose()

    # Get the inferred parameters and uncertainties
    b = mc.vparams[0]
    db = mc.vuncertainties[0]
    a = mc.hparams[0]
    da = mc.huncertainties[0]
    sigma_y = 4.7/np.array(b)
    dsigma_y = 4.7*np.array(db)/(np.array(b)**2)
    gamma_c = 4.7/np.array(a)
    dgamma_c = 4.7*np.array(da)/(np.array(a)**2)
    sigma_y_tc = np.array([4.7, 7.5, 10, 13.1, 18.4, 24.9, 30.4])
    dsigma_y_tc = np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.5])
    gamma_c_tc = np.array([4.7, 6.3, 7.2, 8.0, 9.1, 9.8, 10.9])
    dgamma_c_tc = np.array([0.1, 0.2, 0.2, 0.3, 0.3, 0.7, 0.7])
    fig, ax = plt.subplots(1,1)
    ax.plot(phi, sigma_y_tc, 'b--')
    ax.plot(phi, gamma_c_tc, 'r--')
    ax.errorbar(phi, sigma_y, yerr=dsigma_y, color="blue", linestyle="none", marker="^")
    ax.errorbar(phi, gamma_c, yerr=dgamma_c, color="red", linestyle="none", marker="o")

    # Plot the master curve after shifting the reference to correspond with
    # the yield stress and critical shear rate at phi = 0.68
    mc.change_ref(0.68, 4.7, 4.7)
    fig1, ax1, fig2, ax2, fig3, ax3 = mc.plot(log=True, colormap=plt.cm.viridis)
    ax2.tick_params(which="both",direction="in",top=True,right=True)
    ax3.tick_params(which="both",direction="in",top=True,right=True)

    # Repeat for the viscosity determined from TC fits
    eta_min = 0.037
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

    # Get the inferred parameters and uncertainties
    b = mc.vparams[0]
    db = mc.vuncertainties[0]
    a = mc.hparams[0]
    da = mc.huncertainties[0]
    sigma_y = 4.7/np.array(b)
    dsigma_y = 4.7*np.array(db)/(np.array(b)**2)
    dsigma_y[1:] = sigma_y[1:]*np.sqrt((dsigma_y[1:]/sigma_y[1:])**2 + (dsigma_y[:-1]/sigma_y[:-1])**2)
    gamma_c = 4.7/np.array(a)
    dgamma_c = 4.7*np.array(da)/(np.array(a)**2)
    dgamma_c[1:] = gamma_c[1:]*np.sqrt((dgamma_c[1:]/gamma_c[1:])**2 + (dgamma_c[:-1]/gamma_c[:-1])**2)
    ax.errorbar(phi, sigma_y, yerr=dsigma_y, color="blue", linestyle="none", marker="^", fillstyle="none")
    ax.errorbar(phi, gamma_c, yerr=dgamma_c, color="red", linestyle="none", marker="o", fillstyle="none")
    ax.tick_params(which="both",direction="in",top=True,right=True)
    plt.show()

