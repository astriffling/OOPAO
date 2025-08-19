# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 13:22:10 2022

@author: cheritier -- astriffl
"""

import numpy as np
from OOPAO.Zernike import Zernike
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
from OOPAO.tools.tools import OopaoError

class NCPA:

    def __init__(self,
                 tel,
                 dm,
                 atm,
                 modal_basis='KL',
                 coefficients=None,
                 f2=None,
                 seed=5,
                 M2C=None):
        """
        ************************** REQUIRED PARAMETERS **************************

        An NCPA object consists in defining the 2D map that acts as a static OPD offset. It requires the following parameters
        _ tel              : the telescope object that contains all the informations (diameter, pupil, etc)
        _ dm               : the deformable mirror object that contains all the informations (nAct, misregistrations, etc)
        _ atm              : the atmosphere object that contains all the informations (layers, r0, windspeed, etc)

        ************************** OPTIONAL PARAMETERS **************************

        _ modal_basis            : str, 'KL' (default), 'Zernike'
        _ coefficients           : a list of coefficients of chosen modal basis. The coefficients are normalized to 1 m.
        _ f2                     : a list of 4 elements [amplitude, start mode, end mode, cutoff_freq] which will follow 1/f2 law
        _ seed                   : pseudo-random value to create the NCPA with repeatability
        _ M2C                    : M2C matrix to compute modal basis, override modal_basis entry

        ************************** MAIN PROPERTIES **************************

        The main properties of a NCPA object are listed here:
        _ NCPA.OPD       : the optical path difference map

        ************************** EXEMPLE **************************

        1) Create a blank NCPA object corresponding to a Telescope object
        ncpa = NCPA(tel,dm,atm)

        2) Update the OPD of the NCPA object using a given OPD_map
        ncpa.OPD = OPD_map

        2) Create a source object in H band with a magnitude 8 and combine it to the telescope
        src = Source(optBand = 'H', magnitude = 8)
        src*tel

        3) Create an NCPA object corresponding based on a linear combinaison of modal coefficients(Zernike or KL)
        list_coefficients = [0,0,10e-9,20e-9]
        ncpa = NCPA(tel,dm,atm,coefficients = list_coefficients)                  --> NCPA based on KL modes
        ncpa = NCPA(tel,dm,atm,modal_basis='Zernike',coefficients = list_coefficients) --> NCPA based on Zernikes modes

        4) Create an NCPA following an 1/f2 distribution law on KL modes amplitudes
        ncpa = NCPA(tel,dm,atm,f2=[200e-9,5,25,1])  --> 200 nm RMS NCPA as an 1/f2 law of modes 5 to 25 with a cutoff frequency of 1
        """
        self.basis = modal_basis
        self.tel = tel
        self.atm = atm
        self.dm = dm
        self.seed = seed
        self.M2C = M2C
        if self.M2C is not None:
            self.basis = 'M2C'

        if f2 is None:
            if coefficients is None:
                self.OPD = self.tel.pupil.astype(float)

            if coefficients is not None:
                if type(coefficients) is list:
                    if self.basis == 'KL':
                        self.B = self.KL_basis()[:, :, :len(coefficients)]

                    elif self.basis == 'Zernike':
                        n_max = len(coefficients)
                        self.B = self.Zernike_basis(n_max)
                        
                    elif self.basis == 'M2C':
                        if self.M2C is not None:
                            self.B = self.M2C_basis(self.M2C)
                else:
                    raise OopaoError(
                        'The coefficients should be input as a list.')
                self.OPD = np.matmul(self.B[:,:,:len(coefficients)], np.asarray(coefficients))

        else:
            self.NCPA_f2_law(f2)

        self.tag = 'NCPA'
        print(self)

    def NCPA_f2_law(self, f2):
        if type(f2) is list and len(f2) == 4:
            if self.basis == 'KL': 
                self.B = self.KL_basis()
                
            elif self.basis == 'Zernike':
                self.B = self.Zernike_basis(f2[2])
            
            elif self.basis == 'M2C':
                if self.M2C is not None:
                    self.B = self.M2C_basis(self.M2C)
            else:
                raise OopaoError(f"Basis '{self.basis}' not supported")
            self.OPD = self.f2_law(f2)
            
        else:
            raise OopaoError(
                'f2 should be a list containing [amplitude, start_mode, end_mode, cutoff]')

    def f2_law(self, f2):
        phase = np.sum([np.random.RandomState(i*self.seed).randn()/np.sqrt(
            i+f2[3])*self.B[:, :, i] for i in range(f2[1], f2[2])], axis=0)
        OPD = phase / np.std(phase[np.where(self.tel.pupil == 1)]) * f2[0]
        return OPD
    
    def KL_basis(self):
        M2C_KL = compute_KL_basis(self.tel, self.atm, self.dm, lim=1e-2)
        self.dm.coefs = M2C_KL
        self.tel*self.dm
        B = self.tel.OPD
        return B

    def Zernike_basis(self, n_max):
        self.Z = Zernike(self.tel, J=n_max)
        self.Z.computeZernike(self.tel)
        B = self.Z.modesFullRes
        return B

    def M2C_basis(self, M2C):
        self.dm.coefs = M2C
        self.tel*self.dm
        B = self.tel.OPD
        return B

    def properties(self) -> dict:
        self.prop = dict()
        self.prop['basis']     = f"{'Modal basis':<20s}|{self.basis:^9s}"
        self.prop['amplitude'] = f"{'Amplitude [nm RMS]':<20s}|{np.std(self.OPD[np.where(self.tel.pupil > 0)])*1e9:^9.1f}"
        return self.prop

    def __repr__(self):
        self.properties()
        str_prop = str()
        n_char = len(max(self.prop.values(), key=len))
        for i in range(len(self.prop.values())):
            str_prop += list(self.prop.values())[i] + '\n'
        title = f'\n{" NCPA ":-^{n_char}}\n'
        end_line = f'{"":-^{n_char}}\n'
        table = title + str_prop + end_line
        return table