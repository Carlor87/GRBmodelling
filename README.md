# GRB modelling using NAIMA toolbox
This repository contains a small class and some methods that allow to model 
SSC emission from GRBs using NAIMA.

The class can be easily modified to change the injected electron distribution 
or some of the basic assumptions on the geometry and the doppler boosting.

See the description in the source file where all the methods are described.

A proper document describing the rationale behind the model is in preparation.
A preliminary description can be found
in [this](https://www.overleaf.com/read/ddhndqcfgzxc) overleaf document.

Just to make sure that all the scripts are running properly, here the [file](grbmodel_environment.yml)
that you can use to create a **conda** environment that will automatically 
install all the external dependencies needed to run the code with the same setup
used by the developer.

To create the conda environment type in your terminal:

```bash
conda env create -f grbmodel_environment.yml
```

This will create a conda environment called `grbmodel`.

## Content

### General purpose code

* A file called `grbloader.py` with the class and various methods to model a generic 
GRB with a single zone SSC model.
* An example script called `grbmodel_example.py` that use the data of the GRB190114C as extracted from 
[this](https://ui.adsabs.harvard.edu/abs/2019Natur.575..459M/abstract) paper.
* A datafile named `magic_int1_points.txt` with the datapoints of GRB190114C as extracted from the paper.

## Quick HOW TO
This is based on the general script working on the MAGIC data of the GRB190114C

### Run the code
Following the example script, the first step is to import all the needed modules like
```python
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table,hstack
import astropy.units as u
from astropy.io import ascii
import naima

from grbloader import *
```
and in the last line we import our GRB class.

After this we set some physical parameters, we read the datapoints and create an astropy table
for them with the following:
```python
Eiso = 8e53    # erg
density = 0.5  # cm-3
redshift = 0.4245 
tstart = 68    # s
tstop = 110    # s

tab = ascii.read("magic_int1_points.txt")
newt = Table([tab['energy'], tab['flux'], tab['flux_hi']-tab['flux'], tab['flux']-tab['flux_lo']],
             names=['energy', 'flux', 'flux_error_hi', 'flux_error_lo'])
```
Now we have all that is needed to initialise the GRB class and this is done with
```python
magicgrb = GRBModelling(Eiso, density, [newt], tstart, tstop, redshift,
                        [np.log10(0.07), -1.53, 2.87, 0.45, 0.01],
                        ['log10(eta_e)', 'log10(Ebreak)', 'Index2', 'log10(Ec)', 'log10(B)'],
                        scenario='ISM',
                        cooling_constrain=False)
```
where the second line is the list of the initial parameters, the third line is the list
of the parameters labels, then to use the `ISM` scenario, and the last one is to tell the script
to not add an additional prior based on the cooling time at the break of the electron distributions.

The parameter `scenario` controls the profile of the density of the circumburst material in the
initialization. The possiblities are:
 * `ISM` : constant density of the material as given by the `density` entry
 * `Wind` : density of the material that follows a r^-2 law. If this scenario is chosen, then two additional
   parameters must be given. `mass_loss` that indicates the mass loss rate of the progenitor star (in solar
   masses per year) and `wind_speed` that is the value of the speed of the stellar wind (in km/s).
   In this case the variable `density` will be computed internally according to the values of the
   radius, mass loss rate and wind speed.
 * `average` : that uses a constant density but a derivation of the size and width of the shock that is an averge
   of the values for the ISM and Wind case. 

The fit can then be run by simply calling the function:
`magicgrb.run_naima("testmagic_etae", 128, 50, 100, 2, prefit=True)`

where the arguments are:
* basename of the file it is going to be saved with the whole chain and results
* number of parallel walkers
* steps for the burn-in phase
* number of steps of the chain
* number of processors to use
* option to run a Maximum Likelihood (ML) fit before the MCMC.

_Note: Beware that it can happen that the ML fit converges towards a solution that is not allowed
by the priors on the parameters._

### Plot the results
To plot the results we can use the built-in functions in NAIMA like `plot_chain`, `plot_corner`
and so on, but to plot the final SED, we can also implement our own steps to have more flexibility.

First we get the median of all the parameter distributions and we can use it as the reference 
fit result and we can pass it back to the class
```python
pars = [np.median(a) for a in testrun.flatchain.T]
magicgrb.pars = pars
```
then we can plot the model and the data as
```python
newene = Table([np.logspace(-1, 13, 500)*u.eV], names=['energy'])
naima.plot_data(magicgrb.dataset)
model = magicgrb.naimamodel(magicgrb.pars, newene)[0]
plt.loglog(newene, model, 'k-', label="TOT", lw=3, alpha=0.3)
plt.loglog(newene, magicgrb.synch_comp, 'k--', alpha=0.5, label="Synch.")
plt.loglog(newene, magicgrb.ic_comp, 'k-.', alpha=0.5, label="IC no abs.")
plt.loglog(newene, magicgrb.ic_compGG, 'k:', alpha=0.5, label="IC abs. method 1")
```

If we want also the confidence intervals calculated by NAIMA, we can use the following
```python
newene = [1e-4*u.eV, 1e13*u.eV]
a,b = naima.plot._calc_CI(testrun,confs=[1], modelidx=0, e_range=newene)  # this is a protected naima function...I know...
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
```
