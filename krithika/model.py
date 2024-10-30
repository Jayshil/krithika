import jax.numpy as jnp
from pathlib import Path
import os
from utils import *


class load(object):
    """To load the data
    We will closely follow the juliet philosophy
    We are populating this class based on need-to bases. Will make it rich as needed"""
    def __init__(self, t_lc=None, f_lc=None, fle_lc=None, t_rv=None, f_rv=None, fle_rv=None,\
                 priors=None, GP_regressors_lc=None, GP_regressors_rv=None, LIN_regressors_lc=None,\
                 LIN_regressors_rv=None, output_folder=None, input_folder=None, verbose=False):
        
        # Initialising everything
        self.t_lc = t_lc
        self.f_lc = f_lc
        self.fle_lc = fle_lc
        self.GP_regressors_lc = GP_regressors_lc
        self.LIN_regressors_lc = LIN_regressors_lc

        self.t_rv = t_rv
        self.f_rv = f_rv
        self.fle_rv = fle_rv
        self.GP_regressors_rv = GP_regressors_rv
        self.LIN_regressors_rv = LIN_regressors_rv

        self.out_dir = output_folder
        # Creating output directory if it doesn't exist already
        if not Path(self.out_dir).exists():
            os.mkdir(Path(self.out_dir))
        
        self.in_dir = input_folder

        self.priors = priors

        self.verbose = verbose
        
        # Determining if the fit tpye is lc or rv for each instruments
        if self.t_lc is not None:
            self.lc_instruments = [key for key in self.t_lc.keys()]
            self.ninstruments_lc = len(self.lc_instruments)
        elif self.t_rv is not None:
            self.rv_instruments = [key for key in self.t_rv.keys()]
            self.ninstruments_rv = len(self.rv_instruments)

        # We will read priors to see what kind of fit is to be done for each instruments
        self.data_dict = self.data_dictionary(self.priors)

        # Save the data
        self.save_data()

    def save_data(self):
        """This function is to save the data"""
        # First save the light curves
        if self.t_lc is not None:
            flc = open(self.out_dir + '/lc.dat', 'w')
            for ins in range(self.ninstruments_lc):
                t1, f1, fe1 = self.t_lc[self.lc_instruments[ins]], self.f_lc[self.lc_instruments[ins]], self.fle_lc[self.lc_instruments[ins]]
                for dpt in range(len(t1)):
                    flc.write(str(t1[dpt]) + '\t' + str(f1[dpt]) + '\t' + str(fe1[dpt]) + '\t' + self.lc_instruments[ins] + '\n')
            flc.close()

        # And GP regressors (lc) if exists
        if self.GP_regressors_lc is not None:
            fgplc = open(self.out_dir + '/GP_regressors_lc.dat', 'w')
            for gpkey in self.GP_regressors_lc.keys():
                for dpt in range(len(self.GP_regressors_lc[gpkey])):
                    fgplc.write(str(self.GP_regressors_lc[gpkey][dpt]) + '\t' + gpkey + '\n')
            fgplc.close()
        
        # And LM regressors (lc) if exists
        if self.LIN_regressors_lc is not None:
            flinlc = open(self.out_dir + '/LIN_regressors_lc.dat', 'w')
            for linkey in self.LIN_regressors_lc.keys():
                lin1 = self.LIN_regressors_lc[linkey]
                for dpt in range(lin1.shape[0]):
                    flinlc.write(str(lin1[dpt,:])[1:-1].replace(' ', '\t') + '\t' + linkey + '\n')
            flinlc.close()

        # And RV data
        if self.t_rv is not None:
            frv = open(self.out_dir + '/rv.dat', 'w')
            for ins in range(self.ninstruments_rv):
                t1, f1, fe1 = self.t_rv[self.rv_instruments[ins]], self.f_rv[self.rv_instruments[ins]], self.fle_rv[self.rv_instruments[ins]]
                for dpt in range(len(t1)):
                    frv.write(str(t1[dpt]) + '\t' + str(f1[dpt]) + '\t' + str(fe1[dpt]) + '\t' + self.rv_instruments[ins] + '\n')
            frv.close()

        # GP regressors (rv)
        if self.GP_regressors_rv is not None:
            fgprv = open(self.out_dir + '/GP_regressors_rv.dat', 'w')
            for gpkey in self.GP_regressors_rv.keys():
                for dpt in range(len(self.GP_regressors_rv[gpkey])):
                    fgprv.write(str(self.GP_regressors_rv[gpkey][dpt]) + '\t' + gpkey + '\n')
            fgprv.close()

        # LIN regressors (rv)
        if self.LIN_regressors_rv is not None:
            flinrv = open(self.out_dir + '/LIN_regressors_rv.dat', 'w')
            for linkey in self.LIN_regressors_rv.keys():
                lin1 = self.LIN_regressors_rv[linkey]
                for dpt in range(lin1.shape[0]):
                    flinrv.write(str(lin1[dpt,:])[1:-1].replace(' ', '\t') + '\t' + linkey + '\n')
            flinrv.close()

    def data_dictionary(self, priors):
        # This function will read priors and decide what kind of fit is to be performed on LC data
        # And if there is any GP detrending to be done on the data
        data_dict = {}
        for ins in range(self.ninstruments_lc):
            # Setting some fittypes
            data_dict[self.lc_instruments[ins]]['TransitFit'] = False          # For transit only fit
            data_dict[self.lc_instruments[ins]]['EclipseFit'] = False          # For eclipse only fit
            data_dict[self.lc_instruments[ins]]['TraEclFit'] = False           # For Transit + Eclipse fit (without any phase curve, i.e., phase variation are assumed to be fitted by other models)
            data_dict[self.lc_instruments[ins]]['SinPCFit'] = False            # Transit, Eclipse and a simple sinusoidal phase variation
            data_dict[self.lc_instruments[ins]]['KelpReflPCFit'] = False       # Transit, Eclipse and Kelp reflective light phase curve
            data_dict[self.lc_instruments[ins]]['KelpThmPCFit'] = False        # Transit, Eclipse and Kelp thermal phase curve fit
            data_dict[self.lc_instruments[ins]]['KelpJointPCFit'] = False      # Transit, Eclipse and Kelpt reflective + thermal phase curve fit
            for p in priors.keys():
                vec = p.split('_')

                if p[0:2] == 'q1':
                    if ins in p.split('_'):
                        data_dict[self.lc_instruments[ins]]['TransitFit'] = True
                        if self.verbose:
                            print('>>>> --- Transit fit detected for instrument ', self.lc_instruments[ins])
                
                elif p[0:2] == 'fp':
                    if len(vec) == 2:
                        data_dict[self.lc_instruments[ins]]['EclipseFit'] = True
                        if self.verbose:
                            print('>>>> --- Eclipse fit detected for instrument ', self.lc_instruments[ins])
                    else:
                        if ins in vec:
                            data_dict[self.lc_instruments[ins]]['EclipseFit'] = True
                            if self.verbose:
                                print('>>>> --- Eclipse fit detected for instrument ', self.lc_instruments[ins])
                
                elif p[0:11] == 'phaseoffset':
                    if len(vec) == 2:
                        data_dict[self.lc_instruments[ins]]['SinPCFit'] = True
                        if self.verbose:
                            print('>>>> --- Phase curve (sinusoidal) fit detected for instrument ', self.lc_instruments[ins])
                    else:
                        if ins in vec:
                            data_dict[self.lc_instruments[ins]]['SinPCFit'] = True
                            if self.verbose:
                                print('>>>> --- Phase curve (sinusoidal) fit detected for instrument ', self.lc_instruments[ins])
                
                elif p[0:10] == 'hotspotoff':
                    if len(vec) == 2:
                        data_dict[self.lc_instruments[ins]]['KelpThmPCFit'] = True
                        if self.verbose:
                            print('>>>> --- Phase curve (kelp thermal PC) fit detected for instrument ', self.lc_instruments[ins])
                    else:
                        if ins in vec:
                            data_dict[self.lc_instruments[ins]]['KelpThmPCFit'] = True
                            if self.verbose:
                                print('>>>> --- Phase curve (kelp thermal pc) fit detected for instrument ', self.lc_instruments[ins])
                
                elif p[0:1] == 'g':
                    if len(vec) == 2:
                        data_dict[self.lc_instruments[ins]]['KelpReflPCFit'] = True
                        if self.verbose:
                            print('>>>> --- Phase curve (kelp reflective pc) fit detected for instrument ', self.lc_instruments[ins])
                    else:
                        if ins in vec:
                            data_dict[self.lc_instruments[ins]]['KelpReflPCFit'] = True
                            if self.verbose:
                                print('>>>> --- Phase curve (kelp reflective pc) fit detected for instrument ', self.lc_instruments[ins])
        
            if data_dict[self.lc_instruments[ins]]['TransitFit'] and data_dict[self.lc_instruments[ins]]['EclipseFit']:
                data_dict[self.lc_instruments[ins]]['TransitFit'] = False
                data_dict[self.lc_instruments[ins]]['EclipseFit'] = False
                data_dict[self.lc_instruments[ins]]['TraEclFit'] = True
