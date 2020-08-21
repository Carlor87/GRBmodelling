import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table, hstack
import astropy.units as u
from astropy.io import ascii
import naima
import time

from grbloader import *

tab = ascii.read("magic_int1_points.txt")
newt = Table([tab['energy'], tab['flux'], tab['flux_hi']-tab['flux'], tab['flux']-tab['flux_lo']],
             names=['energy', 'flux', 'flux_error_hi', 'flux_error_lo'])

Eiso = 8e53   # erg
density = 0.5  # cm-3
redshift = 0.4245
tstart = 68   # s
tstop = 110   # s

start_time = time.time()
magicgrb = GRBModelling(8e53, 0.5, [newt], 68, 110, 0.4245,
                        [-1.64541059, -1.74167436,  3.13222916,  1.16770817,  0.37490981],
                        ['log10(eta_e)', 'log10(Ebreak)', 'Index2', 'log10(Ec)', 'log10(B)'],
                        cooling_constrain=False)

# naima.InteractiveModelFitter(magicgrb.naimamodel,magicgrb.pars,data = magicgrb.dataset,labels=magicgrb.labels)

# testrun = magicgrb.run_naima("quickrun", 32, 10, 50, 2, prefit=True)
print("--- %s seconds ---" % (time.time() - start_time))
testrun = naima.read_run("quickrun_chain.h5", magicgrb.naimamodel)  # to reload a previous fit

pars = [np.median(a) for a in testrun.flatchain.T]
magicgrb.pars = pars

newene = Table([np.logspace(-1, 13, 500)*u.eV], names=['energy'])
naima.plot_data(magicgrb.dataset)
model = magicgrb.naimamodel(magicgrb.pars, newene)[0]
plt.loglog(newene, model, 'k-', label="TOT", lw=3, alpha=0.3)
plt.loglog(newene, magicgrb.synch_comp, 'k--', alpha=0.5, label="Synch.")
plt.loglog(newene, magicgrb.ic_comp, 'k-.', alpha=0.5, label="IC no abs.")
plt.loglog(newene, magicgrb.ic_compGG, 'k:', alpha=0.5, label="IC abs. method 1")

newene = [1e-4*u.eV, 1e13*u.eV]
a, b = naima.plot._calc_CI(testrun, confs=[1], modelidx=0, e_range=newene)
xval = a.value
ymax1 = b[0][1].value  # 1 sigma
ymin1 = b[0][0].value  # 1 sigma
plt.fill_between(xval,
                 ymax1,
                 ymin1,
                 alpha=0.2,
                 color='C0',
                 label="1$\sigma$")

plt.ylim(0.9e-9, 1.1e-7)
plt.xlim(1e3, 1e13)
plt.xlabel("Energy [eV]", size=13)
plt.ylabel("$E^2\mathrm{d}N/\mathrm{d}E$ [$\mathrm{erg\,s^{-1}\,cm^{-2}}$]", size=13)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=12)
plt.legend()

plt.show()
