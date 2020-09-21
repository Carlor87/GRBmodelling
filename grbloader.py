"""
# GRB modelling with NAIMA
#
# Author: C. Romoli - MPIK (carlor87 AT gmail.com)
#
# Thanks to F.Aharonian, A.Taylor, D. Khangulyan
# that helped with the theoretical framework
#
# Class and functions to model GRB using the NAIMA package.
#
# References:
# - HESS Collaboration, 2020 - paper on the GRB190829A VHE emission. - work in progress
# - Eungwanichayapant, A. & Aharonian, F., 2009 https://ui.adsabs.harvard.edu/abs/2009IJMPD..18..911E/abstract
# - Aharonian, 2004 - https://ui.adsabs.harvard.edu/abs/2004vhec.book.....A/abstract
# - Aharonian, 2000 - https://ui.adsabs.harvard.edu/abs/2000NewA....5..377A/abstract
# - Atoyan & Aharonian, 1996 - http://adsabs.harvard.edu/abs/1996MNRAS.278..525A
# - Rybicki & Lightman, 1979 - https://ui.adsabs.harvard.edu/abs/1979rpa..book.....R/abstract
"""

import numpy as np

import astropy
from astropy.table import Table
import astropy.units as u
import astropy.constants as con
from astropy.cosmology import WMAP9 as cosmo

import naima
from naima.models import Synchrotron, InverseCompton, ExponentialCutoffBrokenPowerLaw
from naima import uniform_prior, normal_prior

import matplotlib.pyplot as plt

# static variables:
#
m_e = con.m_e.cgs.value
c = con.c.cgs.value
mec2_eV = (con.m_e * con.c ** 2.).to('eV').value
h = con.h.cgs.value
el = con.e.gauss.value
erg_to_eV = 624150912588.3258  # conversion from erg to eV
sigma_T = con.sigma_T.cgs.value
mpc2 = (con.m_p * con.c ** 2.).to('eV')
mpc2_erg = mpc2.to('erg').value


# static methods:
#
# Functions for calculation of gamma-gamma absorption

def sigma_gammagamma(Eph1, Eph2):
    """
    gamma-gamma cross section averaged over scattering angle
    The value is returned in cm2

    Equation 5) from Eungwanichayapant, A.; Aharonian, F., 2009
    (originally from Aharonian, 2004) Approximation good within 3%

    Parameters
    ----------
       Eph1 : array_like
         numpy array of energy of gamma ray in eV
       Eph2 : array_like
         np.array of energy of target photon in eV
    Returns
    -------
        cross_section : astropy.quantity
          angle average cross section for gamma gamma absorption
    """

    CMene = Eph1 * Eph2 / (mec2_eV * mec2_eV)
    mask = CMene > 1.  # mask condition to account for the threshold effect.
    res = np.full(CMene.shape, 0.)
    res[mask] = 3. / (2. * CMene[mask] * CMene[mask]) * sigma_T * \
                     ((CMene[mask] + 0.5 * np.log(CMene[mask]) - 1. / 6. + 1. / (2. * CMene[mask]))
                         * np.log(np.sqrt(CMene[mask]) + np.sqrt(CMene[mask] - 1.)) -
                         (CMene[mask] + 4. / 9. - 1. / (9. * CMene[mask])) *
                         np.sqrt(1. - (1. / CMene[mask])))
    cross_section = res * u.cm * u.cm
    return cross_section


def absorption_coeff(egamma, targetene, target):
    """
    Returns the absorption coefficient K that will then be
    spatially integrated.

    K(E) = \int_e sigma_gg(E,e) * dn/de * de

    where E is the gamma ray energy, e is the energy of the target photon and dn/de is the number distribution
    of the target photon field. (Inner integral of equation 3.24 of Aharonian, 2004)

    Parameters
    ----------
      egamma : array_like
        Energy of the gamma ray photon as astropy unit quantity. Format e.g. [1.]*u.TeV
      targetene : array_like
        energy array of the photon distribution e.g. [1.,2.]*u.eV
      target : array_like
        dnde value of the target photon distribution as 1/(eV cm3)
    Returns
    -------
      abs_coeff : astropy.quantity
        absorption coefficient as astropy quantity (should have units 1/u.cm)
    """

    product = sigma_gammagamma(np.vstack(egamma.to('eV')),
                               targetene.to('eV').value) * target  # make sure the units are correct
    abs_coeff = naima.utils.trapz_loglog(product, targetene, axis=1)
    return abs_coeff


def tau_val(Egamma, targetene, target, size):
    """
    Optical depth assuming homogeneous radiation field.

    From equation 3.24 of Aharonian, 2004 with the assumption of homogeneous photon field.

    Parameters
    ----------
      Egamma    : array_like
        Energy of the gamma ray photon as astropy unit quantity. Format e.g. [1.]*u.TeV
      targetene : array_like
        energy array of the photon distribution e.g. [1.,2.]*u.eV
      target    : array_like
        dnde value of the target photon distribution as 1/(eV cm3)
      size      : astropy.quantity
        size of the integration length as astropy spatial quantity (normally units of cm)
    Returns
    -------
      tau : array_like
        optical depth
    """

    coeff = absorption_coeff(Egamma, targetene, target)
    tau = size.to('cm') * coeff
    return tau


def cutoff_limit(bfield):
    """
     Account for the maximum energy of particles
     for synchrotron emission. Due to the effect of the synchrotron burn-off
     due to the balancing of acceleration and losses of the particle in magnetic field.
     Expression 18 from Aharonian, 2000

    Parameters
    ----------
      bfield : float
        Magnetic field intensity to be given in units of Gauss
    Returns
    -------
      cutoff_ev : float
        log10 of the cutoff energy in units of TeV
    """

    eff = 1.  # acceleration efficiency parameter (eff >= 1)
    cutoff = ((3. / 2.) ** (3. / 4.) *
              np.sqrt(1. / (el ** 3. * bfield)) * (m_e ** 2. * c ** 4.)) * eff ** (-0.5) * u.erg
    cutoff_TeV = (cutoff.value * erg_to_eV * 1e-12)
    return np.log10(cutoff_TeV)


def synch_cooltime_partene(bfield, partene):
    """
    Computes the cooling time for an electron with energy 'partene' in
    Bfield. Returns in units of seconds
    Equation 1 from Aharonian, 2000

    Parameters
    ----------
       bfield : astropy.quantity
         magnetic field as astropy quantity (u.G)
       partene : astropy.quantity
         particle energy as astropy quantity (u.eV)
    Returns
    -------
       tcool : astropy.quantity
         Synchrotron cooling time as astropy quantity (u.s)
    """

    bf = bfield.to('G').value
    epar = partene.to('erg').value
    tcool = (6. * np.pi * m_e ** 4. * c ** 3.) / (sigma_T * m_e ** 2. * epar * bf ** 2.)
    return tcool * u.s


def synch_charene(bfield, partene):
    """
    Function to return
    characteristic energy of synchrotron spectrum

    Equation 3.30 from Aharonian, 2004 (adapted for electrons)

    Parameters
    ----------
       bfield : astropy.quantity
         magnetic field as astropy quantity (u.G)
       partene : astropy.quantity
         particle energy as astropy quantity (u.eV)
    Returns
    -------
       charene : astropy.quantity
         synchrotron characteristic energy as astropy quantity (u.eV)
    """

    bf = bfield.to('G').value
    epar = partene.to('erg').value
    charene = np.sqrt(3. / 2.) * (h * el * bf) / \
                     (2. * np.pi * (m_e ** 3. * c ** 5.)) * epar ** 2.  # in ergs
    return charene * erg_to_eV * u.eV


class GRBModelling:
    """
    Class to produce the grb modelling. The spectral modelling presented here
    is based on the picture of particle acceleration at the forward shock,
    which propagates outwards through the circumburst material
    (see   `R. D. Blandford, C. F. McKee,Physics of Fluids19, 1130 (1976)`).
    Given the total isotropic energy of the explosion
    (`Eiso`), the density of material surrounding the GRB (`n`) and the time of the observation (after trigger),
    it computes the physical parameters of the GRB, like the Lorentz factor `gamma` and the size of the emitting
    shell.

    This class is based on the one used to model the multiwavelength emission of the H.E.S.S. GRB `GRB190829A`.

    Attributes
    ----------
    Eiso : float
        Isotropic energy of the GRB (in units of erg)
    density : float
        density of the circumburst material (in units of cm-3)
    dataset : list of astropy.table.table.Table
        table of observational data. Attribute exists only if a list of tables is passed in the initialization
    tstart : float
        starting time of the observational interval (in units of seconds)
    tstop : float
        stop time of the observational interval (in units of seconds)
    avtime : float
        average time of the observational interval
    redshift : float
        redshift of the GRB
    Dl : astropy.quantity
        luminosity distance of the GRB (as astropy quantity)
    pars : list
        list of parameters of a naima.models.ExponentialCutoffBrokenPowerLaw
    labels : list
        list of parameter names (as strings)
    cooling_constrain : boolean
        If True adds to the prior a constrain for which cooling time at break ~ age of the system. DEFAULT = True
        If synch_nolimit = True, this option does not do anything.
    synch_nolimit : boolean
        False for standard SSC model, True for synchtrotron dominated model. DEFAULT = False
    gamma : float
        Lorentz factor of the GRB at time avtime
    sizer : float
        radius of the expanding shell at time avtime
    shock_energy : astropy.quantity (u.erg)
        available energy in the shock
    Emin : astropy.quantity
        minimum injection energy of the electron distribution
    Wesyn : astropy.quantity
        total energy in the electron distribution
    eta_e : float
        fraction of available energy ending in the electron distribution
    eta_b : float
        fraction of available energy ending in magnetic field energy density
    synch_comp : numpy.array
        synchrotron component of the emission
    ic_comp : numpy.array
        inverse compton component of the emission
    synch_compGG : numpy.array
        synchrotron component of the emission
        with gamma gamma absorption included METHOD 1
    ic_compGG : numpy.array
        inverse compton component of the emission
        with gammagamma absorption included METHOD 1
    synch_compGG2 : numpy.array
        synchrotron component of the emission
        with gamma gamma absorption included METHOD 2
    ic_compGG2 : numpy.array
        inverse compton component of the emission
        with gammagamma absorption included METHOD 2
    naimamodel : bound method
        bound method to the model function
        associated with function load_model_and_prior()
    lnprior : bound method
        bound method to the prior function
        associated with function load_model_and_prior()
    """

    def __init__(self, eiso, dens, data, tstart, tstop, redshift, pars, labels,
                 cooling_constrain=True,
                 synch_nolimit=False):
        """
        Class initialization

        Parameters
        ----------
          eiso : float
            Isotropic energy of the gamma ray burst (given in erg)
          dens : float
            density of the circumburst material (given in cm-3)
          data : list
            list of astropy table with the obs. data. Optional, theoretical line can be computed anyway
          tstart : float
            start time of the observational interval (given in seconds after trigger)
          tstop : float
            stop time of the observational interval (given in seconds after trigger)
          redshift : float
            redshift of the GRB
          pars : list
            list of parameters passed to the model function
          labels : list
            names of the parameters passed to the model
          cooling_constrain : bool
            boolean to add a contrain on cooling time at break ~ age of of the system in the prior function
          synch_nolimit : bool
            boolean to select the synchrotron dominated model
        """

        if isinstance(data, list):
            if all(isinstance(x, astropy.table.table.Table) for x in data):
                self.dataset = data  # dataset astropy table
            else:
                print("WARNING: Not all the elements in your list are formatted as astropy tables!")
                print("Not loading the dataset,")
                print("the code can be used only for computation of theoretical curves")
        else:
            print("WARNING: No dataset given,")
            print("the code can be used only for computation of theoretical curves")
        self.Eiso = eiso  # Eiso of the burst
        self.density = dens  # ambient density around the burst units of cm-3
        self.tstart = tstart  # units of s
        self.tstop = tstop  # units of s
        self.avtime = (tstart + tstop) / 2.  # units of s
        self.redshift = redshift
        self.Dl = cosmo.luminosity_distance(redshift)  # luminosity distance with units
        self.pars = pars  # parameters for the fit
        self.labels = labels  # labels of the parameters
        self.cooling_constrain = cooling_constrain  # if True add in the prior a constrain on cooling break
        self.synch_nolimit = synch_nolimit  # boolean for SSC (=0) or synchrotron without cut-off limit model (=1)
        self.gamma = 0  # Gamma factor of the GRB at certain time
        self.sizer = 0  # External radius of the shell
        self.shock_energy = 0  # Available energy in the shock
        self.Emin = 0 * u.eV  # Minimum injection energy for the particle distribution
        self.Wesyn = 0  # Total energy in the electrons
        self.eta_e = 0  # Fraction of thermal energy going into electron energy
        self.eta_b = 0  # Fraction of thermal energy going into magnetic field
        self.synch_comp = 0  # Spectrum of the synchrotron component
        self.ic_comp = 0  # Spectrum of the IC component
        self.naimamodel = 0  # Model used for the fit - initialized in later function
        self.lnprior = 0  # Prior used for the fit - initialized in later function
        self.gammaval()  # Compute Gamma value and size of the region
        self.load_model_and_prior()  # Loads the NAIMA model and the relative prior
        self.esycool = 0  # Characteristic synchrotron energy corresponding to the break energy of the electrons
        self.synchedens = 0  # total energy density of synchrotron photons
        self.synch_compGG = 0  # synchrotron component of the model with gammagamma absorption with METHOD 1
        self.ic_compGG = 0  # inverse compton component of the model with gammagamma absorption with METHOD 1
        self.synch_compGG2 = 0  # synchrotron component of the model with gammagamma absorption with METHOD 2
        self.ic_compGG2 = 0  # inverse compton component of the model with gammagamma absorption with METHOD 2

    def gammaval(self):
        """
        Computes the Lorentz factor and the size of the region
        Expression from Blandford&McKee,1976.

        Gamma^2 = 3 / (4 * pi) * E_iso / (n * m_p * c^2 * R^3)

        The calculation of the radius uses the relation

        R = 6*Gamma^2*(ct) (average between ISM and Wind scenario)

        Time is the average between the tstart and tstop.
        The functions takes automatically the initialization parameters
        """

        self.gamma = (1. / 6.) ** (3. / 8.) * (
                3.0 * self.Eiso / (4.0 * np.pi * self.density * mpc2_erg * ((c * self.avtime) ** 3.0))) ** 0.125
        self.sizer = 6. * c * self.avtime * self.gamma ** 2

    def load_model_and_prior(self):
        """
        Associates the bound methods
        naimamodel and lnprior to the chosen
        model and prior function.

        Modify here if you want to change the model
        or the priors
        """

        self.naimamodel = self._naimamodel_ind1fixed
        # change here for the prior functions
        # For performance it is better to use if statements here to avoid having them in the prior function
        # the prior function is called everytime and it's better if it does not have if statements inside
        if self.synch_nolimit:
            self.lnprior = self._lnprior_ind2free_nolim
        else:
            if self.cooling_constrain:
                self.lnprior = self._lnprior_ind2free_wlim_wcooling_constrain
            else:
                self.lnprior = self._lnprior_ind2free_wlim

    def calc_photon_density(self, Lsy, sizereg):
        """
        This is a thin shell, we use the approximation
        that the radiation is emitted in a region with radius sizereg.
        No correction factor needed because of thin shell.

        Parameters
        ----------
            Lsy : array_like
              emitted photons per second (units of 1/s)
            sizereg : astropy.quantiy
              size of the region as astropy u.cm quantity

        Returns
        -------
          ph_dens : array_like
            Photon density in the considered emission region.
        """

        # uses a sphere approximation but without the correction factor needed for a
        # full sphere (see e.g. Atoyan, Aharonian, 1996)
        # because we are in a thin shell so: n_ph = Lsy / (4 * pi * R^2 * c)
        return Lsy / (
                4. * np.pi * sizereg ** 2. * c * u.cm / u.s)

    def _naimamodel_ind1fixed(self, pars, data):
        """"
        Example set-up of the free parameters for the SSC implementation
        Index1 of the BPL is fixed as Index2 - 1 (cooling break)
        Index2 of the BPL is free
        The minimum energy and the normalization of the electron distribution are derived
        from the parameter eta_e

        Parameters
        ----------
           pars : list
             parameters of the model as list
           data : astropy.table.table.Table
             observational dataset (as astropy table) or
             if interested only in theoretical lines, astropy table
             containing only a column of energy values.
        Returns
        -------
           model : array_like
             values of the model in correspondence of the data
           electron_distribution : tuple
             electron distribution as tuple energy, electron_distribution(energy) in units of erg
        """

        eta_e = 10. ** pars[0]  # parameter 0: fraction of available energy ending in non-thermal electrons
        ebreak = 10. ** pars[1] * u.TeV  # parameter 1: linked to break energy of the electron distribution (as log10)
        alpha1 = pars[2] - 1.  # fixed to be a cooling break
        alpha2 = pars[2]  # parameter 2: high energy index of the ExponentialCutoffBrokenPowerLaw
        e_cutoff = (10. ** pars[3]) * u.TeV  # parameter 3: High energy cutoff of the electron distribution (as log10)
        bfield = 10. ** (pars[4]) * u.G  # parameter 4: Magnetic field (as log10)
        redf = 1. + self.redshift  # redshift factor
        doppler = self.gamma  # assumption of doppler boosting ~ Gamma
        size_reg = self.sizer * u.cm  # size of the region as astropy quantity
        # volume shell where the emission takes place. The factor 9 comes from considering the shock in the ISM
        # Eq. 7 from GRB190829A paper from H.E.S.S. Collaboration
        vol = 4. * np.pi * self.sizer ** 2. * (
                    self.sizer / (9. * self.gamma))
        shock_energy = 2. * self.gamma ** 2. * self.density * mpc2_erg * u.erg  # available energy in the shock
        eemax = e_cutoff.value * 1e13  # maximum energy of the electron distribution, based on 10 * cut-off value in eV
        self.shock_energy = shock_energy
        self.eta_e = eta_e
        # ratio between magnetic field energy and shock energy
        self.eta_b = (bfield.value ** 2 / (np.pi * 8.)) / shock_energy.value
        ampl = 1. / u.eV  # temporary amplitude
        ECBPL = ExponentialCutoffBrokenPowerLaw(ampl, 1. * u.TeV, ebreak, alpha1, alpha2,
                                                e_cutoff)  # initialization of the electron distribution
        rat = eta_e * self.gamma * mpc2
        ener = np.logspace(9, np.log10(eemax), 100) * u.eV
        eldis = ECBPL(ener)
        ra = naima.utils.trapz_loglog(ener * eldis, ener) / naima.utils.trapz_loglog(eldis, ener)
        emin = rat / ra * 1e9 * u.eV  # calculation of the minimum injection energy. See detailed model explanation
        self.Emin = emin
        SYN = Synchrotron(ECBPL, B=bfield, Eemin=emin, Eemax=eemax * u.eV, nEed=20)
        # TODO: it might need an exception handling in the following line
        amplitude = ((eta_e * shock_energy * vol) / SYN.compute_We(Eemin=emin, Eemax=eemax * u.eV)) / u.eV
        ECBPL = ExponentialCutoffBrokenPowerLaw(amplitude, 1. * u.TeV, ebreak, alpha1, alpha2, e_cutoff)
        SYN = Synchrotron(ECBPL, B=bfield, Eemin=emin, Eemax=eemax * u.eV, nEed=20)
        self.Wesyn = SYN.compute_We(Eemin=emin,
                                    Eemax=eemax * u.eV)  # Computation of the total energy in the electron distribution
        # energy array to compute the target photon number density to compute IC radiation and gamma-gamma absorption
        # characteristic energy at the electron cutoff
        cutoff_charene = np.log10((synch_charene(bfield, e_cutoff)).value)
        min_synch_ene = -4  # minimum energy to start sampling the synchrotron spectrum
        bins_per_decade = 20  # 20 bins per decade to sample the synchrotron spectrum
        bins = int((cutoff_charene - min_synch_ene) * bins_per_decade)
        Esy = np.logspace(min_synch_ene, cutoff_charene + 1, bins) * u.eV
        Lsy = SYN.flux(Esy, distance=0 * u.cm)  # number of synchrotron photons per energy per time (units of 1/eV/s)
        # number density of synchrotron photons (dn/dE) units of 1/eV/cm3
        phn_sy = self.calc_photon_density(Lsy, size_reg)
        self.esycool = (synch_charene(bfield, ebreak))
        self.synchedens = naima.utils.trapz_loglog(Esy * phn_sy, Esy, axis=0).to('erg / cm3')
        # initialization of the IC component
        IC = InverseCompton(ECBPL, seed_photon_fields=[['SSC', Esy, phn_sy]], Eemin=emin, Eemax=eemax * u.eV, nEed=20)
        # Compute the Synchrotron component
        self.synch_comp = (doppler ** 2.) * SYN.sed(data['energy'] / doppler * redf, distance=self.Dl)
        # Compute the IC component
        self.ic_comp = (doppler ** 2.) * IC.sed(data['energy'] / doppler * redf, distance=self.Dl)
        # model = (self.synch_comp+self.ic_comp) # Total model without absorption

        # Compute the optical depth in a shell of width R/(9*Gamma) after transformation of
        # the gamma ray energy of the data in the grb frame
        tauval = tau_val(data['energy'] / doppler * redf, Esy, phn_sy, self.sizer / (9 * self.gamma) * u.cm)
        # Absorption calculation with thickness of shell is R/(9Gamma) for ISM scenario. METHOD 1
        self.synch_compGG = self.synch_comp * np.exp(-tauval)
        self.ic_compGG = self.ic_comp * np.exp(-tauval)
        # model = (self.synch_compGG + self.ic_compGG) # Total model after absorption with METHOD 1

        # Absorption calculation that takes into account the fact that the gamma rays are produced
        # in the same region where they are absorbed. See Rybicki & Lightman eq. 1.29 - 1.30
        # with thickness of the shell R/(9Gamma) for ISM scenario. METHOD 2
        mask = tauval > 1.0e-4
        self.synch_compGG2 = self.synch_comp.copy()
        self.ic_compGG2 = self.ic_comp.copy()
        self.synch_compGG2[mask] = self.synch_comp[mask] / (tauval[mask]) * (1. - np.exp(-tauval[mask]))
        self.ic_compGG2[mask] = self.ic_comp[mask] / (tauval[mask]) * (1. - np.exp(-tauval[mask]))
        model = (self.synch_compGG2 + self.ic_compGG2)  # Total model after absorption with METHOD 2

        ener = np.logspace(np.log10(emin.to('GeV').value), 8,
                           500) * u.GeV  # Energy range to save the electron distribution from emin to 10^8 GeV
        eldis = ECBPL(ener)  # Compute the electron distribution
        electron_distribution = (ener, eldis)
        return model, electron_distribution  # returns model and electron distribution

    def naimamodel_iccomps(self, pars, data, intervals):
        """
        Example set-up of the free parameters for the SSC implementation
        dividing the contribution of the various Synchrotron parts.

        Parameters
        ----------
           pars : list
             parameters of the model as list
           data : astropy.table.table.Table
             observational dataset (as astropy table) or
             if interested only in theoretical lines, astropy table
             containing only a column of energy values.
           intervals : int
             number of intervals to divide the synchrotron component
        Returns
        -------
           icsedl : list
             list of IC components in the SED
        """

        eta_e = 10. ** pars[0]  # parameter 0: fraction of available energy ending in non-thermal electrons
        ebreak = 10. ** pars[1] * u.TeV  # parameter 1: linked to break energy of the electron distribution (as log10)
        alpha1 = pars[2] - 1.  # fixed to be a cooling break
        alpha2 = pars[2]  # parameter 2: high energy index of the ExponentialCutoffBrokenPowerLaw
        e_cutoff = (10. ** pars[3]) * u.TeV  # parameter 3: High energy cutoff of the electron distribution (as log10)
        bfield = 10. ** (pars[4]) * u.G  # parameter 4: Magnetic field (as log10)
        redf = 1. + self.redshift  # redshift factor
        doppler = self.gamma  # assumption of doppler boosting ~ Gamma
        size_reg = self.sizer * u.cm  # size of the region as astropy quantity
        # volume shell where the emission takes place. The factor 9 comes from considering the shock in the ISM
        # Eq. 7 from GRB190829A paper from H.E.S.S. Collaboration
        vol = 4. * np.pi * self.sizer ** 2. * (
                self.sizer / (9. * self.gamma))
        shock_energy = 2. * self.gamma ** 2. * self.density * mpc2.value * u.erg  # available energy in the shock
        eemax = e_cutoff.value * 1e13
        self.eta_e = eta_e
        self.eta_b = (bfield.value ** 2 / (np.pi * 8.)) / shock_energy.value
        ampl = 1. / u.eV  # temporary amplitude
        ECBPL = ExponentialCutoffBrokenPowerLaw(ampl, 1. * u.TeV, ebreak, alpha1, alpha2,
                                                e_cutoff)  # initialization of the electron distribution
        rat = eta_e * self.gamma * mpc2
        ener = np.logspace(9, np.log10(eemax), 100) * u.eV
        eldis = ECBPL(ener)
        ra = naima.utils.trapz_loglog(ener * eldis, ener) / naima.utils.trapz_loglog(eldis, ener)
        emin = rat / ra * 1e9 * u.eV  # calculation of the minimum injection energy. See detailed model explanation
        self.Emin = emin
        SYN = Synchrotron(ECBPL, B=bfield, Eemin=emin, Eemax=eemax * u.eV, nEed=20)
        amplitude = ((eta_e * shock_energy * vol) / SYN.compute_We(Eemin=emin, Eemax=eemax * u.eV)) / u.eV
        ECBPL = ExponentialCutoffBrokenPowerLaw(amplitude, 1. * u.TeV, ebreak, alpha1, alpha2, e_cutoff)
        SYN = Synchrotron(ECBPL, B=bfield, Eemin=emin, Eemax=eemax * u.eV, nEed=20)
        ener = np.linspace(-6, 2, intervals)
        Esyl = []
        Lsyl = []
        phn_syl = []
        ICl = []
        icsedl = []
        for i in range(intervals - 1):
            Esylc = np.logspace(ener[i], ener[i + 1], 100) * u.eV
            Esyl.append(Esylc)
            print("synch energy: ", ener[i], ener[i + 1])
            Lsylc = SYN.flux(Esylc, distance=0 * u.cm)
            Lsyl.append(Lsylc)
            phn_sylc = self.calc_photon_density(Lsylc, size_reg)
            phn_syl.append(phn_sylc)
            name = "SSC%i" % i
            IClc = InverseCompton(ECBPL, seed_photon_fields=[[name, Esylc, phn_sylc]], Eemin=emin, Eemax=eemax * u.eV,
                                  nEed=80)
            ICl.append(IClc)
            icsedlc = doppler ** 2 * IClc.sed(data['energy'] / doppler * redf, distance=self.Dl)
            icsedl.append(icsedlc)
        return icsedl

    def _lnprior_ind2free_wlim(self, pars):
        """
        Basic prior function where some basic parameters of the electron distribution are left free.
        The following parameters are free to vary:

           - pars[0] = log10(eta_e) (limits: [-5,0])
           - pars[1] = log10(break energy), in Tev (limits: [-6,Eblim]) where Eblim = 1 (could made more complex)
           - pars[2] = high energy index, on linear scale (limits: [-1,5])
           - pars[3] = log10(cut-off energy), in TeV (limits: [Eblim,cutoff_limit(Bfield)])
             where cutoff_limit(Bfield) is the cut-off dictated by the synchrotron burn-off limit
           - pars[4] = log10(magnetic field), in Gauss (limits: [-3,1])

        Parameters
        ----------
          pars : list
            list of parameters passed to the model
        Returns
        -------
          prior : float
            prior probability
        """

        Eblim = 1.  # can be substituted with a proper limit based on the dataset at hand
        # rest of the prior
        # lower limit of the break energy (pars[1]) is the minimum injection energy.
        prior0 = uniform_prior(pars[0], -5, 0)
        prior1 = uniform_prior(pars[1], -6, Eblim)
        prior2 = uniform_prior(pars[2], -1, 5)
        prior3 = uniform_prior(pars[3], Eblim, cutoff_limit(10 ** pars[4]))
        prior4 = uniform_prior(pars[4], -3, 1)
        lnprior = prior0 + prior1 + prior2 + prior3 + prior4
        return lnprior

    def _lnprior_ind2free_wlim_wcooling_constrain(self, pars):
        """
        Basic prior function where some basic parameters of the electron distribution are left free.
        The following parameters are free to vary:

           - pars[0] = log10(eta_e) (limits: [-5,0])
           - pars[1] = log10(break energy), in Tev (limits: [-6,Eblim]) where Eblim = 1 (could made more complex)
           - pars[2] = high energy index, on linear scale (limits: [-1,5])
           - pars[3] = log10(cut-off energy), in TeV (limits: [Eblim,cutoff_limit(Bfield)])
             where cutoff_limit(Bfield) is the cut-off dictated by the synchrotron burn-off limit
           - pars[4] = log10(magnetic field), in Gauss (limits: [-3,1])

        In this function there is an additional prior given by having the synchrotron cooling time of
        an electron at the break to be equal to the comoving age of the system. This prior is implemented
        through a normal prior distribution.

        Parameters
        ----------
          pars : list
            list of parameters passed to the model
        Returns
        -------
          prior : float
            prior probability
        """

        tcool = self.avtime * self.gamma  # age of system in comoving frame
        # cooling time at break energy
        datacool = synch_cooltime_partene(10 ** pars[4] * u.G, 10. ** pars[1] * u.TeV).value
        # gaussian prior on the cooling time at break ~ age of system
        additional = normal_prior(datacool, tcool, tcool * 0.5)
        Eblim = 1.  # can be substituted with a proper limit based on the dataset at hand
        # rest of the prior
        # lower limit of the break energy (pars[1]) is the minimum injection energy.
        prior0 = uniform_prior(pars[0], -5, 0)
        prior1 = uniform_prior(pars[1], -6, Eblim)
        prior2 = uniform_prior(pars[2], -1, 5)
        prior3 = uniform_prior(pars[3], Eblim, cutoff_limit(10 ** pars[4]))
        prior4 = uniform_prior(pars[4], -3, 1)
        lnprior = prior0 + prior1 + prior2 + prior3 + prior4 + additional
        return lnprior

    def _lnprior_ind2free_nolim(self, pars):
        """
        Basic prior function where some basic parameters of the electron distribution are left free.
        In this function the cut-off is not limited by the synchrotron burn-off limit
        The following parameters are free to vary:

           - pars[0] = log10(eta_e) (limits: [-5,0])
           - pars[1] = log10(break energy), in Tev (limits: [-6,Eblim]) where Eblim = 1
           - pars[2] = high energy index, on linear scale (limits: [-1,5])
           - pars[3] = log10(cut-off energy), in TeV (limits: [-3,7])
           - pars[4] = log10(magnetic field), in Gauss (limits: [-3,1])

        Parameters
        ----------
          pars : list
            list of parameters passed to the model
        Returns
        -------
          prior : float
            prior probability
        """

        Eblim = 1.  # can be substituted with a proper limit based on the dataset at hand
        # rest of the prior
        # lower limit of the break energy (pars[1]) is the minimum injection energy.
        prior0 = uniform_prior(pars[0], -5, 0)
        prior1 = uniform_prior(pars[1], np.log10(self.Emin.to('TeV').value), Eblim)
        prior2 = uniform_prior(pars[2], -1, 5)
        prior3 = uniform_prior(pars[3], -3, 7)
        prior4 = uniform_prior(pars[4], -3, 1)
        additional = 0  # no additional gaussian prior
        lnprior = prior0 + prior1 + prior2 + prior3 + prior4 + additional
        return lnprior

    def get_Benergydensity(self):
        """
        Returns the magnetic field energy density in cgs system
        """

        bedens = (10. ** self.pars[4]) ** 2. / (8. * np.pi)  # free parameter 4 is the B field
        return bedens * u.erg / u.cm / u.cm / u.cm

    def get_Eltotenergy(self):
        """
        Returns total energy in the electron distribution
        """

        return self.Wesyn  # which is the total electron energy injected

    def run_naima(self, filename, nwalkers, nburn, nrun, threads, prefit=True):
        """
        run the naima fitting routine. Wrapper around naima.run_sampler, naima.save_run,
        and naima.save_results_table.
        Filename is the basename of the file to be saved.
        Default arguments are to run a prefit to the dataset using a ML approach.
        Beware that the ML fit might converge in a region not allowed by the parameter space

        Parameters
        ----------
          filename : string
            string with the base name of the file to be saved
          nwalkers : int
            number of parallel walkers used for the MCMC fitting
          nburn : int
            number of burn-in steps
          nrun : int
            number of steps after burn-in
          threads : int
            number of parallel threads
          prefit : bool
            If `True` performs a Maximum Likelihood fit to get a better starting point for
            for the MCMC chain (Default = True)
        Returns
        -------
          sampler : array_like
            full chain of the MCMC
        """

        sampler, pos = naima.run_sampler(data_table=self.dataset,
                                         p0=self.pars,
                                         labels=self.labels,
                                         model=self.naimamodel,
                                         prior=self.lnprior,
                                         prefit=prefit, guess=False,
                                         nwalkers=nwalkers, nburn=nburn, nrun=nrun, threads=threads)
        naima.save_run(filename=filename, sampler=sampler, clobber=True)
        naima.save_results_table(outname=filename, sampler=sampler)
        return sampler

    def integral_flux(self, emin, emax, energyflux):
        """
        Compute the integral flux (or energy flux) of of the model between emin and emax.

        Parameters
        ----------
          emin : float
            minimum energy of the interval (in eV)
          emax : float
            maximum energy of the interval (in eV)
          energyflux : bool
            boolean set to True to compute energy flux (erg/cm2/s) False for normal flux (1/cm2/s)
        Returns
        -------
          intflux : astropy.quantity
            integral flux (in units of 1/cm2/s or erg/cm2/s)
        """

        enarray = (np.logspace(np.log10(emin), np.log10(emax), 10) * u.eV).to('erg')  # in erg
        newene = Table([enarray], names=['energy'])
        model = self.naimamodel(self.pars, newene)[0]
        if energyflux:
            ednde = model / enarray
            intflux = naima.utils.trapz_loglog(ednde, enarray)
        else:
            dnde = model / enarray / enarray
            intflux = naima.utils.trapz_loglog(dnde, enarray)
        return intflux

    def quick_plot_sed(self, emin, emax, ymin, ymax):
        """
        Function for a quick plot of the model on a user specific energy range.
        If a dataset is present, this is plotted using NAIMA internal routine.

        Parameters
        ----------
          emin : float
            minimum energy of the interval (in eV)
          emax : float
            maximum energy of the interval (in eV)
          ymin : float
            minimum value for the y-axis (in erg/cm2/s)
          ymax : float
            maximum value for the y-axis (in erg/cm2/s)
        """

        bins = int(np.log10(emax/emin) * 20.)  # use 20 bins per decade
        newene = Table([np.logspace(np.log10(emin), np.log10(emax), bins) * u.eV], names=['energy'])  # energy in eV
        model = self.naimamodel(self.pars, newene)  # make sure we are computing the model for the new energy range
        f = plt.figure()
        if self.dataset:
            naima.plot_data(self.dataset, figure=f)
        plt.loglog(newene, model[0], 'k-', label='TOTAL', lw=3, alpha=0.8)
        plt.loglog(newene, self.synch_comp, 'k--', label='Synch. w/o abs.')
        plt.loglog(newene, self.ic_comp, 'k:', label='IC w/o abs.')
        plt.legend()
        plt.xlim(emin, emax)
        plt.ylim(ymin, ymax)
