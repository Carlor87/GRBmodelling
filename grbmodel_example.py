"""
Example script to run the GRBModelling class
on the data from the GRB190114C detected by the MAGIC Collaboration

Author: Carlo Romoli (MPIK)
"""

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table, hstack
import astropy.units as u
from astropy.io import ascii
import naima

from grbloader import *


def main():
    # Open the file and format the table in order to be properly read by the script
    tab = ascii.read("magic_int1_points.txt")
    newt = Table([tab['energy'], tab['flux'], tab['flux_hi']-tab['flux'], tab['flux']-tab['flux_lo']],
                 names=['energy', 'flux', 'flux_error_hi', 'flux_error_lo'])

    Eiso = 8e53   # erg
    density = 0.5  # cm-3
    redshift = 0.4245
    tstart = 68   # s
    tstop = 110   # s

    """
    Initialize the GRBModelling class with the data and the given parameters
    """
    magicgrb = GRBModelling(8e53, 0.5, [newt], 68, 110, 0.4245,
                            [-1.44, -1.62,  3.17,  1.32,  0.29],
                            ['log10(eta_e)', 'log10(Ebreak)', 'Index2', 'log10(Ec)', 'log10(B)'],
                            scenario='ISM',
                            cooling_constrain=False)

    """ 
    Load the NAIMA interactive fitter (optional) 
    """
    # naima.InteractiveModelFitter(magicgrb.naimamodel,magicgrb.pars,data = magicgrb.dataset,labels=magicgrb.labels)

    """ 
    Run the fitting routine 
    """
    testrun = magicgrb.run_naima("quickrun", 64, 50, 100, 2, prefit=False)
    # testrun = naima.read_run("quickrun_chain.h5", magicgrb.naimamodel)  # to reload a previous fit using NAIMA function

    pars = [np.mean(a) for a in testrun.flatchain.T]  # read the parameter distributions and save the median
    magicgrb.pars = pars

    """
    Load and plot the results
    """
    newene = Table([np.logspace(-1, 13, 500)*u.eV], names=['energy'])  # define a new energy axis
    naima.plot_data(magicgrb.dataset)  # plot the dataset
    model = magicgrb.naimamodel(magicgrb.pars, newene)[0]  # load the model curve for the new energy axis
    plt.loglog(newene, model, 'k-', label="TOT", lw=3, alpha=0.3)
    plt.loglog(newene, magicgrb.synch_comp, 'k--', alpha=0.5, label="Synch.")
    plt.loglog(newene, magicgrb.ic_comp, 'k-.', alpha=0.5, label="IC no abs.")
    plt.loglog(newene, magicgrb.ic_compGG, 'k:', alpha=0.5, label="IC abs. method 1")

    """
    Compute and plot the confidence interval
    Here a protected member of naima is used. This allows to compute the
    confidence interval for an arbitrary energy range (see NAIMA documentation)
    """
    newene = [1e-4*u.eV, 1e13*u.eV]
    a, b = naima.plot._calc_CI(testrun, confs=[1], modelidx=0, e_range=newene)  # access the confidence interval directly
    xval = a.value
    ymax1 = b[0][1].value  # 1 sigma
    ymin1 = b[0][0].value  # 1 sigma
    plt.fill_between(xval,
                     ymax1,
                     ymin1,
                     alpha=0.2,
                     color='C0',
                     label="1$\sigma$")  # plot

    plt.ylim(0.9e-9, 1.1e-7)
    plt.xlim(1e3, 1e13)
    plt.xlabel("Energy [eV]", size=13)
    plt.ylabel("$E^2\mathrm{d}N/\mathrm{d}E$ [$\mathrm{erg\,s^{-1}\,cm^{-2}}$]", size=13)
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=12)
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()