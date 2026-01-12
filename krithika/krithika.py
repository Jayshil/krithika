import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gd
from astroquery.mast import Observations
from astropy.io import fits
from astropy.table import Table
from astropy.stats import mad_std
from scipy.interpolate import interp1d
from pathlib import Path
from glob import glob
from tqdm import tqdm
import plotstyles
import pickle
import juliet
import utils
import warnings
import os

try:
    from photutils.aperture import CircularAnnulus, CircularAperture, ApertureStats
    from photutils.aperture import aperture_photometry as aphot
except:
    print('Warning: photutils is not installed! Will not be able to use photutils based photometry.')


class CHEOPSData(object):
    """
    To manipulate CHEOPS data, from simple reading to extraction.
    """
    def __init__(self, filename):
        self.fname = filename
    
    def pipe_data(self, bgmin=300, not_flg_arr=None):
        """
        Function to read PIPE data for CHEOPS

        This function will read PIPE data with removing flagged data, hihgh background data etc.
        
        Parameters:
        -----------
        f1 : str
            Name (along with location) of the fits file
        bgmin : int, float
            Threshold for background; all points with higher backgrounds
            will be discarded.
            Default is 300 e-/pix
        not_flg_arr : ndarray
            Array containing the other non-zero flags to include while assembling the data.
            Default is None.
        
        Returns:
        --------
        data : dict
            Dictionary containing BJD time, normalized flux with
            errors on it, roll angle, xc, yc, BG, thermFront2 and
            principal components of PSF fitting (U0 to Un)
        """
        hdul = fits.open(self.fname)
        tab = Table.read(hdul[1])
        # Masking datasets
        flg = np.asarray(tab['FLAG'])                 # Flagged data
        msk = np.where(flg==0)[0]                     # Creating a mask to remove flg>0 values
        if not_flg_arr is not None:
            for i in range(len(not_flg_arr)):
                msk_f1 = np.where(flg==not_flg_arr[i])[0]
                msk = np.hstack((msk, msk_f1))
        # Gathering dataset
        Us_n = np.array([])
        Us = []
        for i in tab.colnames:
            if i[0] == 'U':
                Us_n = np.hstack((Us_n, i))
        for j in range(len(Us_n)):
            usn = np.asarray(tab[Us_n[j]])[msk]
            Us.append(usn)
        tim, flx, flxe = np.asarray(tab['BJD_TIME'])[msk], np.asarray(tab['FLUX'])[msk], np.asarray(tab['FLUXERR'])[msk]
        roll, xc, yc, bg = np.asarray(tab['ROLL'])[msk], np.asarray(tab['XC'])[msk], np.asarray(tab['YC'])[msk], np.asarray(tab['BG'])[msk]
        tf2 = np.asarray(tab['thermFront_2'])[msk]
        # Masking those points with high background values
        msk1 = np.where(bg<bgmin)[0]
        tim, flx, flxe, roll, xc, yc, bg, tf2 = tim[msk1], flx[msk1], flxe[msk1], roll[msk1], xc[msk1], yc[msk1], bg[msk1], tf2[msk1]
        Us1 = []
        for i in range(len(Us_n)):
            us1 = Us[i][msk1]
            Us1.append(us1)
        # Normalising flux
        flx, flxe = flx/np.median(flx), flxe/np.median(flx)
        data = {}
        data['TIME'], data['FLUX'], data['FLUX_ERR'] = tim, flx, flxe
        data['ROLL'], data['XC'], data['YC'], data['BG'] = roll, xc, yc, bg
        data['TF2'] = tf2
        for i in range(len(Us_n)):
            data[Us_n[i]] = Us1[i]
        return data
    
class julietPlots(object):
    """Plotting helper for results produced by a `juliet` analysis.

    This class loads a `juliet` input/output folder, computes models from the
    fitted posterior and provides a collection of convenience plotting and
    data-processing methods for light-curve visualisation. Typical usage is
    to instantiate the class with the `input_folder` used for a juliet fit
    and then call plotting helpers such as :meth:`full_model_lc`,
    :meth:`plot_gp` or :meth:`detrend_data`.

    Attributes
    ----------
    input_folder : str, optional
        Path to the juliet input/output folder provided at construction.
        The user need to provide either input_folder or dataset and res
    dataset : juliet.load object, optioanl
        The object returned by ``juliet.load`` for the given input folder.
    res : juliet.fit, optional
        The result of calling ``dataset.fit(...)``.
    N : int
        Number of posterior samples drawn when evaluating sampled models.
    **kwargs : dict, optional
        Any additional keywords provided to juliet.fit object.

    Main methods
    ------------
    full_model_lc(instruments=None, save=False, nrandom=50, quantile_models=True)
        Plot observed light curves with the full fitted model and residuals.
    
    phase_folded_lc(phmin=0.8, instruments=None, highres=False, ...)
        Plot phase folder light curve with best-fitted planetary model for one or more instruments.
    
    plot_gp(instruments=None, highres=False, one_plot=False, ...)
        Plot GP components and binned residuals for one or more instruments.

    plot_fake_allan_deviation(instruments=None, binmax=10, method='pipe', timeunit=None)
        Plot "allan deviation" plot, which is noise as a function of binning, for one or more instruments
    
    plot_corner(planet_only=False, save=True)
        Plot corner plots for fitted posterior samples, can use planet-only or all parameters.
    
    """
    def __init__(self, input_folder=None, dataset=None, res=None, N=1000, **kwargs):
        # First order of business: loading the juliet folder
        # We can optinally provide datasets and res directly

        if ( dataset is None ) and ( input_folder is not None ):
            self.dataset = juliet.load(input_folder=input_folder)
            self.res = self.dataset.fit(**kwargs)
        else:
            self.dataset = dataset
            self.res = res

        # Saving those kwargs
        self.fit_kwargs = kwargs

        # Location of the input folder
        self.input_folder = input_folder

        # Number of samples
        self.nsamps = N

        # Computing all models, for all instruments
        self.models_all_ins = {}
        self.all_mods_ins = {}
        # Sorting indices for all instruments (sometimes GP regressors are not time -- in that case, times 
        # are sorted according to GP regressors, which will mess-up the time array -- we need to sort it again)
        self.idx_time_sort = {}
        for i in range(len(self.dataset.inames_lc)):
            # Evaluating the models
            ## This will return 5 elements: model samples, median model, upper_CI, lower_CI, and components
            self.models_all_ins[self.dataset.inames_lc[i]] =\
                self.res.lc.evaluate(self.dataset.inames_lc[i], nsamples=self.nsamps, return_err=True,\
                                     return_components=True, return_samples=True)

            # Saving all models
            self.all_mods_ins[self.dataset.inames_lc[i]] = self.res.lc.model[self.dataset.inames_lc[i]]

            # Sorting array
            self.idx_time_sort[self.dataset.inames_lc[i]] = np.argsort( self.dataset.times_lc[self.dataset.inames_lc[i]] )

    def highres_planet_models_workaround(self, instrument, times_highres, GP_reg_highres, lin_reg_highres):
        """
        The usual way of generating high time-resolution models
        is not working with juliet when light_travel_delay=True for
        occultations and phase curve modelling. This is a workaround for this."""
        
        # Creating dictionaries to save high-res times, fluxes, and errors
        tim_highres, fl_highres, fle_highres = {}, {}, {}
        ## Actual high-res time array
        tim_highres[instrument] = times_highres
        ## For flux and flux error highres array, we will simply use array with ones
        fl_highres[instrument], fle_highres[instrument] = np.ones(len(times_highres)), np.ones(len(times_highres))

        # GP and linear regressors would be arrays, so creating dictionaries to save them
        if GP_reg_highres is not None:
            gp_pars = {}
            gp_pars[instrument] = GP_reg_highres
        else:
            gp_pars = None
        
        if lin_reg_highres is not None:
            lin_pars = {}
            lin_pars[instrument] = lin_reg_highres
        else:
            lin_pars = None

        ## Again, running juliet.load and juliet.fit
        data_highres = juliet.load(priors=self.input_folder + '/priors.dat', t_lc=tim_highres, y_lc=fl_highres, yerr_lc=fle_highres,\
                                   GP_regressors_lc=gp_pars, linear_regressors_lc=lin_pars, out_folder=self.input_folder)
        res_highres = data_highres.fit(**self.fit_kwargs)

        post_samps_new = res_highres.posteriors['posterior_samples'].copy()
        for i in post_samps_new.keys():
            if ( i[0:5] == 'mflux' ) or ( i[0:9] == 'mdilution' ) or ( i[0:5] == 'theta' ) or ( i[0:2] == 'GP' ):
                post_samps_new[i] = np.zeros(len(post_samps_new[i]))

        planet_only_models_highres = \
            res_highres.lc.evaluate(instrument, nsamples=self.nsamps, return_err=True, parameter_values=post_samps_new,\
                                    return_components=True, return_samples=True, evaluate_transit=True)

        return planet_only_models_highres
        

    def full_model_lc(self, instruments=None, save=False, nrandom=50, quantile_models=True):
        """Plot the full fitted light-curve model for one or more instruments.

        This method creates a figure per instrument showing the observed
        light curve with the fitted (full) model overplotted and a
        residuals panel beneath.

        Parameters
        ----------
        instruments : list or None
            List of instrument names to plot. If ``None``, all instruments
            in the current juliet dataset plotted.
        save : bool, optional
            If ``True``, each figure is saved to
            ``<input_folder>/full_model_<instrument>.png``. Default ``False``.
        nrandom : int, optional
            Number of random posterior-sample models to draw when
            ``quantile_models`` is ``False``. Default is 50.
        quantile_models : bool, optional
            If ``True``, plot the 68%% credible interval as a filled band;
            if ``False``, plot ``nrandom`` random posterior models instead.

        Returns
        -------
        all_fig : list
            List of matplotlib Figure objects created (one per instrument).
        all_axs1 : list
            List of top-panel Axes (data + model) for each figure.
        all_axs2 : list
            List of bottom-panel Axes (residuals) for each figure.

        Notes
        -----
        The routine expects that ``self.models_all_ins`` was populated by
        a prior call (see the class constructor) and that the dataset
        arrays (times, data, errors) are available in ``self.dataset``.
        """

        # Computing the model for all instruments
        if instruments is None:
            # If instruments is None, then all instruments are selected
            instruments = self.dataset.inames_lc
        
        all_fig, all_axs1, all_axs2 = [], [], []
        for i in range(len(instruments)):
            # Data
            tim_ins = self.dataset.times_lc[instruments[i]][self.idx_time_sort[instruments[i]]]
            fl_ins = self.dataset.data_lc[instruments[i]][self.idx_time_sort[instruments[i]]]
            fle_ins = self.dataset.errors_lc[instruments[i]][self.idx_time_sort[instruments[i]]]

            ## Median model
            full_model = self.models_all_ins[instruments[i]][1][self.idx_time_sort[instruments[i]]]

            ## Quantile models
            up_68CI = self.models_all_ins[instruments[i]][2][self.idx_time_sort[instruments[i]]]
            lo_68CI = self.models_all_ins[instruments[i]][3][self.idx_time_sort[instruments[i]]]

            ## Random models
            sample_models = self.models_all_ins[instruments[i]][0][:, self.idx_time_sort[instruments[i]]]
            idx = np.random.choice( np.arange(sample_models.shape[0]), size=nrandom )
            random_models = sample_models[idx, :]

            # And figure
            fig = plt.figure()
            gs = gd.GridSpec(2,1, height_ratios=[2,1])

            # Top panel
            ax1 = plt.subplot(gs[0])
            ax1.errorbar(tim_ins, fl_ins, yerr=fle_ins, fmt='.', color='dodgerblue')#, alpha=0.3)
            ax1.plot(tim_ins, full_model, c='navy', lw=2.5, zorder=50)
            if quantile_models:
                ax1.fill_between(x=tim_ins, y1=lo_68CI, y2=up_68CI, color='orangered', alpha=0.5, zorder=25)
            else:
                for rand in range(nrandom):
                    ax1.plot(tim_ins, random_models[rand,:], color='orangered', alpha=0.5, lw=1., zorder=25)

            ax1.set_ylabel('Relative Flux')
            ax1.set_title('Full fitted model for instrument ' + str(instruments[i]))
            ax1.set_xlim(np.min(tim_ins), np.max(tim_ins))
            ax1.xaxis.set_major_formatter(plt.NullFormatter())

            # Bottom panel
            ax2 = plt.subplot(gs[1])
            ax2.errorbar(tim_ins, (fl_ins-full_model)*1e6, yerr=fle_ins*1e6, fmt='.', color='dodgerblue')#, alpha=0.3)
            ax2.axhline(y=0.0, c='black', ls='--', zorder=100)
            ax2.set_ylabel('Residuals [ppm]')
            ax2.set_xlabel('Time [BJD]')
            ax2.set_xlim(np.min(tim_ins), np.max(tim_ins))

            all_fig.append(fig)
            all_axs1.append(ax1)
            all_axs2.append(ax2)

            if save:
                plt.savefig(self.input_folder + '/full_model_' + instruments[i] + '.png', dpi=250)
        
        return all_fig, all_axs1, all_axs2
    
    def detrend_data(self, phmin, instruments):
        """This function will generate detrended data from the raw data. It will also generate phases."""
        
        # First, create detrended dataset and models
        ## For detrended dataset
        self.phase, self.detrended_data, self.detrended_errs = {}, {}, {}
        self.ph_minimum, self.ph_maximum = 1e10, 1e-10

        ## For normalising the data, we will compute F1 and F2,
        ## We can use the same F1 and F2 to normalise the model as well
        ## So, let's save them!
        self.F1, self.F2 = {}, {}

        for i in range(len(instruments)):
            # -----------------------------------------
            #  Our first task is to compute
            #  period and t0:
            # -----------------------------------------
            try:
                per = np.nanmedian(self.res.posteriors['posterior_samples']['P_p1'])
            except:
                per = self.all_mods_ins[instruments[i]]['params'].per
            
            try:
                t0 = np.nanmedian(self.res.posteriors['posterior_samples']['t0_p1'])
            except:
                t0 = self.all_mods_ins[instruments[i]]['params'].t0
            # -----------------------------------------

            # -----------------------------------------
            #  Next we would like to compute GP
            #  and linear models, so that we can 
            #  detrend the data
            # -----------------------------------------
            # GP model
            if instruments[i] in self.dataset.GP_lc_arguments.keys():
                gp_model = self.all_mods_ins[instruments[i]]['GP']
            else:
                gp_model = 0.

            if instruments[i] in self.dataset.lm_lc_arguments.keys():
                linear_model = self.models_all_ins[instruments[i]][-1]['lm']
            else:
                linear_model = 0.
            
            # -----------------------------------------

            # -----------------------------------------
            #  Finally, we want to compute F1 and F2
            #  So we can normalise the detrended data
            #  The full model is defined as (courtesy of the original juliet code),
            #  y = [ T * ( D / (1 + D*M) ) ] + [ (1 - D) / (1 + D*M)]
            #  => y = T * F1 + F2
            #  => T = (y - F2) / F1
            # -----------------------------------------

            mflx = self.res.posteriors['posterior_samples']['mflux_' + instruments[i]]
            try:
                mdil = self.res.posteriors['posterior_samples']['mdilution_' + instruments[i]]
            except:
                mdil = np.ones( len(mflx) )
            
            F1 = np.nanmedian( mdil / ( 1 + (mdil * mflx) ) )
            F2 = np.nanmedian( (1 - mdil) / (1 + (mdil * mflx) ) )

            ## Saving F1 and F2
            self.F1[instruments[i]], self.F2[instruments[i]] = F1, F2

            sigw = np.nanmedian(self.res.posteriors['posterior_samples']['sigma_w_' + instruments[i]] * 1e-6)

            ## Here we create detrended data by subtracting GP (if any) and linear (if any) models from the raw data
            self.detrended_data[instruments[i]] = ( self.dataset.data_lc[instruments[i]] - gp_model - linear_model - F2) / F1 
            self.detrended_errs[instruments[i]] = np.sqrt( self.dataset.errors_lc[instruments[i]]**2 + sigw**2 )

            phases_instrument = juliet.utils.get_phases(self.dataset.times_lc[instruments[i]], P=per, t0=t0, phmin=phmin)
            self.phase[instruments[i]] = phases_instrument
            
            # Computing the minimum and maximum of the orbital phases
            if np.max(phases_instrument) > self.ph_maximum:
                self.ph_maximum = np.max(phases_instrument)
            if np.min(phases_instrument) < self.ph_minimum:
                self.ph_minimum = np.min(phases_instrument)

    
    def detrend_model(self, instruments, phmin, highres):
        # If we want to create high-resolution model (highres=True), then we will use the same times for all instruments
        # If not, then we will use different times for different instruments
        self.times_model, self.phases_model = {}, {}
        self.planet_only_models = {} 
        
        # -----------------------------------------
        #  Our first task is to compute
        #  period and t0:
        # -----------------------------------------
        try:
            per = np.nanmedian(self.res.posteriors['posterior_samples']['P_p1'])
        except:
            per = self.all_mods_ins[instruments[0]]['params'].per
        
        try:
            t0 = np.nanmedian(self.res.posteriors['posterior_samples']['t0_p1'])
        except:
            t0 = self.all_mods_ins[instruments[0]]['params'].t0

        # We also have to create a new posterior array since when we compute detrended model,
        # we only want uncertainties from planetary parameters to be propagated to the model uncertainties
        # So, we will make posteriors of mflux, theta, and GP to zero, such that any uncertainties
        # in them would not propagate to model uncertainties.

        post_samps_new = self.res.posteriors['posterior_samples'].copy()
        for i in post_samps_new.keys():
            if ( i[0:5] == 'mflux' ) or ( i[0:9] == 'mdilution' ) or ( i[0:5] == 'theta' ) or ( i[0:2] == 'GP' ):
                post_samps_new[i] = np.zeros(len(post_samps_new[i]))
        
        if highres:
            ## First selecting the times
            ## This this is a high time resolution scenario so we will create a dummy time array
            dummy_phs_highres = np.linspace(self.ph_minimum, self.ph_maximum, 10000)
            dummy_time_highres = t0 + (dummy_phs_highres * per)

            # Now loop over all instruments to create high time-resolution planet-only models
            for ins in range(len(instruments)):
                
                ## First we have to check if there are GP or linear model fitted to the data
                ## Because, we have to create high-resolution GP and linear regressors as well.
                ## We will do this by interpolating the original GP and linear regressors

                ### High resolution GP regressors
                ### (we have to remember that everything is sorted according to GP regressors)
                ### (while dummy time is sorted according to times)
                ### (we will perform interpolation to arrays sorted according to time)
                if instruments[ins] in self.dataset.GP_lc_arguments.keys():
                    #! Try/except because it is possible that not all instruments have GP model
                    try:
                        inter_gp = interp1d(x=self.dataset.times_lc[instruments[ins]][self.idx_time_sort[instruments[ins]]],\
                                            y=self.dataset.GP_lc_arguments[instruments[ins]][:,0][self.idx_time_sort[instruments[ins]]],\
                                            kind='cubic')
                        gp_reg_highres = inter_gp(x=dummy_time_highres)

                        #### Now, we can sort time, phase, gp regressors according to GP regressors
                        idx_gp = np.argsort( gp_reg_highres )
                        dummy_time_highres, dummy_phs_highres = dummy_time_highres[idx_gp], dummy_phs_highres[idx_gp]
                        gp_reg_highres = gp_reg_highres[idx_gp]
                    except:
                        gp_reg_highres = None
                else:
                    gp_reg_highres = None

                ### High resolution linear regressors
                if instruments[ins] in self.dataset.lm_lc_arguments.keys():
                    #! Again, try/except because not all instruments might have a linear model
                    ## (Here, we don't need to sort according to time -- because the dummy_times should have now
                    ##  sorted according to GP regressors, if applicable)
                    try:
                        inter_lin = interp1d(x=self.dataset.times_lc[instruments[ins]],\
                                             y=self.dataset.lm_lc_arguments[instruments[ins]],\
                                             kind='cubic', axis=0)
                        lin_reg_highres = inter_lin(x=dummy_time_highres)
                    except:
                        lin_reg_highres = None
                else:
                    lin_reg_highres = None

                try:
                    # This high-res time models will not work for occultations/phase curves, when light_travel_delay=True
                    self.planet_only_models[instruments[ins]] = \
                        self.res.lc.evaluate(instruments[ins], nsamples=self.nsamps, return_err=True, parameter_values=post_samps_new,\
                                            return_components=True, return_samples=True, evaluate_transit=True,\
                                            t=dummy_time_highres, GPregressors=gp_reg_highres, LMregressors=lin_reg_highres)
                except:
                    # In case when it doesn't work, I have invented a workaround
                    self.planet_only_models[instruments[ins]] = \
                        self.highres_planet_models_workaround(instrument=instruments[ins], times_highres=dummy_time_highres,\
                                                              GP_reg_highres=gp_reg_highres, lin_reg_highres=lin_reg_highres)
                ## Saving times and phases for model
                self.times_model[instruments[ins]] = dummy_time_highres
                self.phases_model[instruments[ins]] = dummy_phs_highres
        else:
            for ins in range(len(instruments)):
                # This should be fairly trivial since we don't need to compute the models at higher time resolution
                self.planet_only_models[instruments[ins]] = \
                    self.res.lc.evaluate(instruments[ins], nsamples=self.nsamps, return_err=True, parameter_values=post_samps_new,\
                                            return_components=True, return_samples=True, evaluate_transit=True)
                
                ## Saving times and phases for model
                self.times_model[instruments[ins]] = self.dataset.times_lc[instruments[ins]]
                self.phases_model[instruments[ins]] = juliet.utils.get_phases(t=self.dataset.times_lc[instruments[ins]], P=per, t0=t0, phmin=phmin)

    
    def phase_folded_lc(self, phmin=0.8, instruments=None, highres=False, nrandom=50, quantile_models=True, one_plot=None, figsize=(16/1.5, 9/1.5), pycheops_binning=False, nos_bin_tra=20, nos_bin_pc=30):
        """Plot phase-folded light curves and models for the fitted dataset.

        This method computes detrended data and planetary models by
        calling ``detrend_data`` and ``detrend_model`` internally, detects
        whether the fit contains transits, eclipses or phase-curves, and
        then produces matplotlib figures showing the data, model, and
        residuals. It can produce either one figure per instrument or a single
        combined figure with all instruments plotted together.

        Parameters
        ----------
        phmin : float, optional
            Minimum phase value used for generating orbital phase (default
            ``0.8``).
        instruments : list or None, optional
            List of instrument names to plot. If ``None``, all instruments
            in the juliet dataset (``self.dataset.inames_lc``) are used. One plot
            is produced if all instruments are provided; otherwise, one plot
            per instrument is created.
        highres : bool, optional
            If ``True``, compute and plot high time-resolution planet-only
            models. Default ``False``.
        nrandom : int, optional
            Number of random posterior-sample models to draw when
            ``quantile_models`` is ``False``. Default ``50``.
        quantile_models : bool, optional
            If ``True``, display the 68%% credible interval as a filled
            band; if ``False``, overplot ``nrandom`` random posterior
            samples. Default ``True``.
        one_plot : bool or None, optional
            If ``True``, produce a single combined plot for all instruments
            (regardless of the number of instruments provided). If
            ``False``, produce one plot per instrument (unless all
            instruments are provided). If ``None``, the behaviour is as
            just described. Default ``None``.
        figsize : tuple, optional
            Figure size passed to matplotlib when creating figures. Default
            is ``(16/1.5, 9/1.5)``.
        pycheops_binning : bool, optional
            If ``True``, use binning as produced by ``pycheops`` the
            binned datapoints. If ``False``, it will use default ``juliet``
            binning. Default ``False``.
        nos_bin_tra : int, optional
            Number of total binned data points in transit plot. Default is 20.
        nos_bin_pc : int, optional
            Number of total binned data points in phase curve plot. Default is 30.

        Returns
        -------
        If a single combined plot is produced the function returns the
        combined ``fig`` and the primary axes used. If multiple figures are
        generated, the function returns lists: ``figs_all, axs1_all,
        axs2_all, axs3_all, axs4_all`` corresponding to created figures and
        their axes panels.

        """
        # -------------------------------------------
        #         Do we want one plot?
        # -------------------------------------------
        # If instruments == None, then we will make one common plot for _all_ instruments
        # Also, if all instruments are provided, this will make one common plot
        # Otherwise, one plot per instrument
        # However, sometimes we want to force one plot, so we have the option one_plot
        if one_plot is None:
            one_plot = True
            if instruments != None:
                if len(instruments) != len(self.dataset.inames_lc):
                    one_plot = False
            else:
                instruments = self.dataset.inames_lc

        # -------------------------------------------
        #               Compute t14
        # -------------------------------------------
        
        # Let's compute the transit/occultation duration for the planet (that can help setting the xlimits of the plots)
        ## We need several other parameters for that first

        ### Orbital period
        try:
            per = np.nanmedian(self.res.posteriors['posterior_samples']['P_p1'])
        except:
            per = self.all_mods_ins[instruments[0]]['params'].per
        
        ### Rp/R*
        try:
            rprs = np.nanmedian(self.res.posteriors['posterior_samples']['p_p1'])
        except:
            rprs = self.all_mods_ins[instruments[0]]['params'].rp

        ### a/R*
        if 'rho' in self.res.model_parameters:
            ### This would mean that we have fitted for rho. We can convert rho to a/*
            ar = utils.rho_to_ar(np.nanmedian(self.res.posteriors['posterior_samples']['rho']), per)
        else:
            ### That means that we directly fit for a/R*
            try:
                ar = np.nanmedian(self.res.posteriors['posterior_samples']['a_p1'])
            except:
                ar = self.all_mods_ins[instruments[0]]['params'].a
        
        ### b
        try:
            b = np.nanmedian(self.res.posteriors['posterior_samples']['b_p1'])
        except:
            inc = self.all_mods_ins[instruments[0]]['params'].inc
            b = utils.inc_to_b(inc=inc, ar=ar, ecc=self.all_mods_ins[instruments[0]]['params'].ecc,\
                               omega=self.all_mods_ins[instruments[0]]['params'].w)
        t14 = utils.t14(per=per, ar=ar, rprs=rprs, b=b, ecc=self.all_mods_ins[instruments[0]]['params'].ecc,\
                        omega=self.all_mods_ins[instruments[0]]['params'].w)
        t14_phs = t14 / per

        # -------------------------------------------
        #     Computing detrended data and model
        # -------------------------------------------
        self.detrend_data(phmin=phmin, instruments=instruments)
        self.detrend_model(instruments=instruments, phmin=phmin, highres=highres)

        # -------------------------------------------
        #           Type of fitting
        # -------------------------------------------
        # We can find the type of fitting just by looking at the phase range covered
        phasecurve, transit, eclipse = False, False, False

        for ins in range(len(instruments)):
            if self.dataset.lc_options[instruments[ins]]['TranEclFit']:
                phasecurve = True
            elif self.dataset.lc_options[instruments[ins]]['EclipseFit']:
                eclipse = True
            else:
                transit = True
        
        ## For phase curve fitting, eclipse fitting is by default true, so, we need to manually turn it off
        if phasecurve and eclipse:
            eclipse = False
        
        ## Sometimes, even for just eclipse fitting we might have turned full transit+eclipse fitting, so double check
        ## if there is indeed phase curve fitting in the data
        if phasecurve:
            diff_phs = self.ph_maximum - self.ph_minimum
            if diff_phs<0.3:
                phasecurve = False
                eclipse = True

        # -------------------------------------------
        #         Starting figure code:
        #      one plot: do plt.figure() here
        #   Depending on phase curve/transit/occ
        #   We need either 4 panels, or 2 panels
        # -------------------------------------------
        if one_plot:
            if not phasecurve:
                fig = plt.figure(figsize=figsize)
                gs = gd.GridSpec(2,1, height_ratios=[2,1])
                ax1 = plt.subplot(gs[0])  # Top panel
                ax2 = plt.subplot(gs[1])  # Bottom panel
                ax3, ax4 = None, None
            else:
                fig = plt.figure(figsize=figsize)
                gs = gd.GridSpec(2, 2, height_ratios=[2,1], width_ratios=[1,3])#, wspace=0.1)
                ax1 = plt.subplot(gs[0,0])  # Upper-left (for transit data)
                ax2 = plt.subplot(gs[1,0])  # Bottom-left (for transit residuals)

                ax3 = plt.subplot(gs[0,1])  # Upper-right (for phase curve)
                ax4 = plt.subplot(gs[1,1])  # Bottom-right (for phase curve residuals)
            
            # If there is one plot, then we also need to 
            # collect all data to compute binned data
            bin_phs, bin_fl, bin_fle, bin_res = np.array([]), np.array([]), np.array([]), np.array([])
            # Also, in case we don't have highres model, we want to save all models before plotting
            all_models_for_plotting = np.array([])
            all_u68_mods_plotting, all_l68_mods_plotting = np.array([]), np.array([])

        # ------------------------------------------------------------------------
        # Creating a list of all fig, axes to return at the end of the function
        # if one_plot==False
        if not one_plot:
            figs_all, axs1_all, axs2_all, axs3_all, axs4_all = [], [], [], [], []
        
        for i in range(len(instruments)):

            # -------------------------------------------
            #         Starting figure code:
            #    If not one plot: do plt.figure() 
            #           inside the loop
            #   Depending on phase curve/transit/occ
            #   We need either 4 panels, or 2 panels
            # -------------------------------------------

            if not one_plot:
                if not phasecurve:
                    fig = plt.figure(figsize=figsize)
                    gs = gd.GridSpec(2,1, height_ratios=[2,1])
                    ax1 = plt.subplot(gs[0])
                    ax2 = plt.subplot(gs[1])
                    ax3, ax4 = None, None
                else:
                    fig = plt.figure(figsize=figsize)
                    gs = gd.GridSpec(2, 2, height_ratios=[2,1], width_ratios=[1,3])#, wspace=0.1)
                    ax1 = plt.subplot(gs[0,0])  # Upper-left (for transit data)
                    ax2 = plt.subplot(gs[1,0])  # Bottom-left (for transit residuals)

                    ax3 = plt.subplot(gs[0,1])  # Upper-right (for phase curve)
                    ax4 = plt.subplot(gs[1,1])  # Bottom-right (for phase curve residuals)
            
            # -------------------------------------------
            #      Collecting the required data
            # -------------------------------------------
            # Detrended data
            ## Sorting according to phase
            idx_phs = np.argsort(self.phase[instruments[i]])

            phs = self.phase[instruments[i]][idx_phs]
            detrend_data = self.detrended_data[instruments[i]][idx_phs]
            detrend_errs = self.detrended_errs[instruments[i]][idx_phs]

            # Detrended model
            ## Sorting according to phase
            idx_phs_mod = np.argsort( self.phases_model[instruments[i]] )

            phs_model = self.phases_model[instruments[i]][idx_phs_mod]
            detrend_model = self.planet_only_models[instruments[i]][1][idx_phs_mod]

            ## Quantile models
            up_68CI = self.planet_only_models[instruments[i]][2][idx_phs_mod]
            lo_68CI = self.planet_only_models[instruments[i]][3][idx_phs_mod]

            ## Random models
            sample_models = self.planet_only_models[instruments[i]][0][:, idx_phs_mod]
            idx = np.random.choice( np.arange(sample_models.shape[0]), size=nrandom )
            random_models = sample_models[idx, :]

            # Residuals
            residuals = ( self.dataset.data_lc[instruments[i]] - self.models_all_ins[instruments[i]][1] )[idx_phs]

            # If we have phase curve fitting, we will use ppm units for the y-axis
            if phasecurve or eclipse:
                detrend_data, detrend_errs = (detrend_data-1.)*1e6, detrend_errs*1e6
                detrend_model = (detrend_model-1.)*1e6
                random_models = (random_models-1.)*1e6
                lo_68CI, up_68CI = (lo_68CI-1.)*1e6, (up_68CI-1.)*1e6

            if one_plot:
                # -------------------------------------------
                #       Saving the data for binning 
                #            outside the loop
                # -------------------------------------------
                bin_phs = np.hstack(( bin_phs, phs ))
                bin_fl, bin_fle = np.hstack(( bin_fl, detrend_data )), np.hstack(( bin_fle, detrend_errs ))
                bin_res = np.hstack(( bin_res, residuals ))
                all_models_for_plotting = np.hstack(( all_models_for_plotting, detrend_model ))
                all_l68_mods_plotting = np.hstack(( all_l68_mods_plotting, lo_68CI ))
                all_u68_mods_plotting = np.hstack(( all_u68_mods_plotting, up_68CI ))

            # ----------------------------------------------------------
            #     The actual plotting code starts from here
            # ----------------------------------------------------------

            # -------------------------------------------
            #         Top left: transit data
            # -------------------------------------------
            if phasecurve:
                ppt = 1e-3
            else:
                ppt = 1.
            
            ax1.errorbar(phs, detrend_data*ppt, fmt='.', alpha=0.25, color='dodgerblue', zorder=1)
            if (not one_plot) or highres:
                ## if we are making one-plot, we can plot the detrended model out of loop
                ax1.plot(phs_model, detrend_model*ppt, color='navy', lw=2.5, zorder=50)
            if quantile_models:
                if (not one_plot) or highres:
                    ## Again, if we are making one plot, we can plot the detrended quantile models out of loop
                    ax1.fill_between(phs_model, y1=lo_68CI*ppt, y2=up_68CI*ppt, color='orangered', alpha=0.15, zorder=25)
            else:
                for rand in range(nrandom):
                    ax1.plot(phs_model, random_models[rand,:]*ppt, color='orangered', alpha=0.5, lw=1., zorder=25)
            
            # Limits
            ## Maximum of the transit model would always be 1. (or, 0 in case we have phase curve model, and we decided to use ppm)
            if transit or phasecurve:
                ax1.set_xlim(-0.7*t14_phs, 0.7*t14_phs)
            else:
                ax1.set_xlim(0.5-0.7*t14_phs, 0.5+0.7*t14_phs)
            
            if transit:
                ax1.set_ylim([np.min(detrend_model)-0.2*np.ptp(detrend_model), 1+0.2*np.ptp(detrend_model)])
            elif phasecurve:
                ax1.set_ylim([np.min(detrend_model*ppt)-0.2*np.ptp(detrend_model*ppt), 0.+0.2*np.ptp(detrend_model*ppt)])
            else:
                ax1.set_ylim([1-0.3*np.ptp(detrend_model), np.max(detrend_model)+0.3*np.ptp(detrend_model)])

            # -------------------------------------------
            #        Bottom left: transit residuals
            # -------------------------------------------
            ax2.errorbar(phs, residuals*ppt*1e6, fmt='.', alpha=0.25, color='dodgerblue', zorder=1)
            ax2.axhline(y=0., ls='--', color='k', zorder=50)

            # Limits (x-limit same as the top axis; for y-lim, we will set 3-sigma from median)
            if transit or phasecurve:
                ax2.set_xlim(-0.7*t14_phs, 0.7*t14_phs)
            else:
                ax2.set_xlim(0.5-0.7*t14_phs, 0.5+0.7*t14_phs)

            # -------------------------------------------
            #        Binned data: For TRANSIT
            # -------------------------------------------
            ## We do binned data here: ONLY if we don't have one plot
            if not one_plot:
                if transit or phasecurve:
                    idx_transits = ( phs > -0.8*t14_phs ) & ( phs < 0.8*t14_phs )
                else:
                    idx_transits = ( phs > 0.5-0.8*t14_phs ) & ( phs < 0.5+0.8*t14_phs )
                
                if not pycheops_binning:
                    nbin = int( np.sum(idx_transits) / nos_bin_tra )

                    bin_phs_tra, bin_fl_tra, bin_fle_tra = juliet.utils.bin_data(x=phs[idx_transits], y=detrend_data[idx_transits]*ppt, n_bin=nbin, yerr=detrend_errs[idx_transits]*ppt)
                    if phasecurve or eclipse:
                        _, bin_res_tra, bin_reserr_tra = juliet.utils.bin_data(x=phs[idx_transits], y=residuals[idx_transits]*ppt*1e6, n_bin=nbin, yerr=detrend_errs[idx_transits]*ppt)
                    else:
                        # When transit only, we haven't multiplied the errorbars with 1e6 yet; so we need to do it again
                        _, bin_res_tra, bin_reserr_tra = juliet.utils.bin_data(x=phs[idx_transits], y=residuals[idx_transits]*1e6, n_bin=nbin, yerr=detrend_errs[idx_transits]*1e6)
                else:
                    binwid = np.ptp( phs[idx_transits] ) / nos_bin_tra

                    bin_phs_tra, bin_fl_tra, bin_fle_tra, _ = utils.lcbin(time=phs[idx_transits], flux=detrend_data[idx_transits]*ppt, binwidth=binwid)
                    if phasecurve or eclipse:
                        _, bin_res_tra, bin_reserr_tra, _ = utils.lcbin(time=phs[idx_transits], flux=residuals[idx_transits]*ppt*1e6, binwidth=binwid)
                    else:
                        # When transit only, we haven't multiplied the errorbars with 1e6 yet; so we need to do it again
                        _, bin_res_tra, bin_reserr_tra, _ = utils.lcbin(time=phs[idx_transits], flux=residuals[idx_transits]*1e6, binwidth=binwid)

                # Plotting them
                ax1.errorbar(bin_phs_tra, bin_fl_tra, yerr=bin_fle_tra, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)
                ax2.errorbar(bin_phs_tra, bin_res_tra, yerr=bin_reserr_tra, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)

                # Setting y-lim on residuals based on binned residuals
                ax2.set_ylim(np.nanmedian(bin_res_tra)-5*utils.pipe_mad(bin_res_tra), np.nanmedian(bin_res_tra)+5*utils.pipe_mad(bin_res_tra))

            # -------------------------------------------
            #         If phase curve modelling,
            #        then we also need right panel
            # -------------------------------------------
            if phasecurve:
                # -------------------------------------------
                #         Top right: phase curve
                # -------------------------------------------
                ax3.errorbar(phs, detrend_data, fmt='.', alpha=0.25, color='dodgerblue', zorder=1)
                if (not one_plot) or highres:
                    ## If we are making one plot, then we can take this outside of the loop
                    ax3.plot(phs_model, detrend_model, color='navy', lw=2.5, zorder=50)
                ax3.axhline(0.0, color='k', ls='--', lw=1.5, zorder=45)
                
                if quantile_models:
                    if (not one_plot) or highres:
                        ## If we are making one plot, we can take this outside of the loop
                        ax3.fill_between(phs_model, y1=lo_68CI, y2=up_68CI, color='orangered', alpha=0.15, zorder=25)
                else:
                    for rand in range(nrandom):
                        ax3.plot(phs_model, random_models[rand,:], color='orangered', alpha=0.5, lw=1., zorder=25)
                
                # Limits
                ax3.set_xlim(phmin-1., phmin)
                ## For y-lim; we have baseline of 0, we can leave half occultation depth/phase amplitude below and above
                ylen = np.ptp( detrend_model[np.abs(phs_model) > 0.1] )
                ax3.set_ylim([-0.4*ylen, 1.5*ylen])

                # -------------------------------------------
                #     Bottom right: phase curve residuals
                # -------------------------------------------
                ax4.errorbar(phs, residuals*1e6, fmt='.', alpha=0.25, color='dodgerblue', zorder=1)
                ax4.axhline(y=0., ls='--', color='k', zorder=10)

                # Limits
                ax4.set_xlim(phmin-1., phmin)

                # -------------------------------------------
                #        Binned data: For PHASE CURVE
                # -------------------------------------------
                ## We do binned data here: ONLY if we don't have one plot
                if not one_plot:
                    if not pycheops_binning:
                        bin_phs_pc, bin_fl_pc, bin_fle_pc = juliet.utils.bin_data(x=phs, y=detrend_data, n_bin=int( len(phs)/nos_bin_pc ), yerr=detrend_errs)
                        _, bin_res_pc, bin_reserr_pc = juliet.utils.bin_data(x=phs, y=residuals*1e6, n_bin=int( len(phs)/nos_bin_pc ), yerr=detrend_errs)
                    else:
                        bin_phs_pc, bin_fl_pc, bin_fle_pc, _ = utils.lcbin(time=phs, flux=detrend_data, binwidth=1/nos_bin_pc)
                        _, bin_res_pc, bin_reserr_pc, _ = utils.lcbin(time=phs, flux=residuals*1e6, binwidth=1/nos_bin_pc)

                    # Plotting them
                    ax3.errorbar(bin_phs_pc, bin_fl_pc, yerr=bin_fle_pc, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)
                    ax4.errorbar(bin_phs_pc, bin_res_pc, yerr=bin_reserr_pc, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)

                    ax4.set_ylim(np.nanmedian(bin_res_pc)-5*utils.pipe_mad(bin_res_pc), np.nanmedian(bin_res_pc)+5*utils.pipe_mad(bin_res_pc))

            if not one_plot:
                # ---------------------------------------------------
                #           Figure labels etc.
                # ---------------------------------------------------
                # ------------------------
                #.   Transits
                # ------------------------
                ## First, removing the x-axis labels of top panels
                ax1.xaxis.set_major_formatter(plt.NullFormatter())
                # x-and y-labels
                if not phasecurve:
                    if not eclipse:
                        ax1.set_ylabel('Normalised flux')
                    else:
                        ax1.set_ylabel('Normalised flux [ppm]')
                    ax2.set_ylabel('Residuals [ppm]')
                else:
                    ax1.set_ylabel('Normalised flux [ppt]')
                    ax2.set_ylabel('Residuals [ppt]')
                
                ax2.set_xlabel('Orbital phase')

                # ------------------------
                #.   Transits
                # ------------------------
                if phasecurve:
                    ## Removing x-axis labels of top panels
                    ax3.xaxis.set_major_formatter(plt.NullFormatter())

                    ## Now, we will switch the y-axis labels to right side for phase curve panels
                    ax3.yaxis.tick_right()
                    ax3.tick_params(labelright=True)
                    ax3.yaxis.set_label_position('right')
                    
                    ax4.yaxis.tick_right()
                    ax4.tick_params(labelright=True)
                    ax4.yaxis.set_label_position('right')

                    ## x-and y-labels
                    ax3.set_ylabel('Normalised flux [ppm]', rotation=270, labelpad=25)
                    
                    ax4.set_xlabel('Orbital phase')
                    ax4.set_ylabel('Residuals [ppm]', rotation=270, labelpad=25)

                # Saving figs and axes if not one_plot
                figs_all.append(fig)
                axs1_all.append(ax1)
                axs2_all.append(ax2)
                axs3_all.append(ax3)
                axs4_all.append(ax4)

        # -------------------------------------------------------------
        #            Now, we plot the binned data when 
        #                    we have one plot
        # -------------------------------------------------------------
        if one_plot:# and (not highres):
            ## This ppt will convert transit data to PPT, when phase curve is plotted
            if phasecurve:
                ppt = 1e-3
            else:
                ppt = 1.
            
            # -------------------------------------------
            #        Binned data: For Transits
            # -------------------------------------------
            idx_phs_bin = np.argsort(bin_phs)
            bin_phs, bin_fl, bin_fle, bin_res = bin_phs[idx_phs_bin], bin_fl[idx_phs_bin], bin_fle[idx_phs_bin], bin_res[idx_phs_bin]


            # Now, we want to decide nbin (number of data points to bin)
            # We would like to have different binning for transit and phase curve -- so first selecting nbin for transit/occultation
            if transit or phasecurve:
                idx_transits = ( bin_phs > -0.8*t14_phs ) & ( bin_phs < 0.8*t14_phs )
            else:
                idx_transits = ( bin_phs > 0.5-0.8*t14_phs ) & ( bin_phs < 0.5+0.8*t14_phs )
            
            if not pycheops_binning:
                nbin = int( np.sum(idx_transits) / nos_bin_tra )

                # Performing binning
                bin_phs_tra, bin_fl_tra, bin_fle_tra = juliet.utils.bin_data(x=bin_phs[idx_transits], y=bin_fl[idx_transits]*ppt, n_bin=nbin, yerr=bin_fle[idx_transits]*ppt)
                if phasecurve or eclipse:
                    _, bin_res_tra, bin_reserr_tra = juliet.utils.bin_data(x=bin_phs[idx_transits], y=bin_res[idx_transits]*ppt*1e6, n_bin=nbin, yerr=bin_fle[idx_transits]*ppt)
                else:
                    # When transit only, we haven't multiplied the errorbars with 1e6 yet; so we need to do it again
                    _, bin_res_tra, bin_reserr_tra = juliet.utils.bin_data(x=bin_phs[idx_transits], y=bin_res[idx_transits]*1e6, n_bin=nbin, yerr=bin_fle[idx_transits]*1e6)
            else:
                binwid = np.ptp( bin_phs[idx_transits] ) / nos_bin_tra

                # Performing binning
                bin_phs_tra, bin_fl_tra, bin_fle_tra, _ = utils.lcbin(time=bin_phs[idx_transits], flux=bin_fl[idx_transits]*ppt, binwidth=binwid)
                if phasecurve or eclipse:
                    _, bin_res_tra, bin_reserr_tra, _ = utils.lcbin(time=bin_phs[idx_transits], flux=bin_res[idx_transits]*ppt*1e6, binwidth=binwid)
                else:
                    # When transit only, we haven't multiplied the errorbars with 1e6 yet; so we need to do it again
                    _, bin_res_tra, bin_reserr_tra, _ = utils.lcbin(time=bin_phs[idx_transits], flux=bin_res[idx_transits]*1e6, binwidth=binwid)

            # Plotting them
            ax1.errorbar(bin_phs_tra, bin_fl_tra, yerr=bin_fle_tra, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)
            ax2.errorbar(bin_phs_tra, bin_res_tra, yerr=bin_reserr_tra, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)

            # Setting y-lim on residuals based on binned residuals
            ax2.set_ylim(np.nanmedian(bin_res_tra)-5*utils.pipe_mad(bin_res_tra), np.nanmedian(bin_res_tra)+5*utils.pipe_mad(bin_res_tra))
            
            # -------------------------------------------
            #        Binned data: For PHASE CURVE
            # -------------------------------------------
            if phasecurve:
                if not pycheops_binning:
                    bin_phs_pc, bin_fl_pc, bin_fle_pc = juliet.utils.bin_data(x=bin_phs, y=bin_fl, n_bin=int( len(bin_phs)/nos_bin_pc ), yerr=bin_fle)
                    _, bin_res_pc, bin_reserr_pc = juliet.utils.bin_data(x=bin_phs, y=bin_res*1e6, n_bin=int( len(bin_phs)/nos_bin_pc ), yerr=bin_fle)
                else:
                    bin_phs_pc, bin_fl_pc, bin_fle_pc, _ = utils.lcbin(time=bin_phs, flux=bin_fl, binwidth=1/nos_bin_pc)
                    _, bin_res_pc, bin_reserr_pc, _ = utils.lcbin(time=bin_phs, flux=bin_res*1e6, binwidth=1/nos_bin_pc)

                # Plotting them
                ax3.errorbar(bin_phs_pc, bin_fl_pc, yerr=bin_fle_pc, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)
                ax4.errorbar(bin_phs_pc, bin_res_pc, yerr=bin_reserr_pc, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)

                ax4.set_ylim(np.nanmedian(bin_res_pc)-5*utils.pipe_mad(bin_res_pc), np.nanmedian(bin_res_pc)+5*utils.pipe_mad(bin_res_pc))

            
            # -----------------------------------------------------------------------
            #
            #               Now, we want to plot the planetary model
            #                        if highres = False
            #
            # -----------------------------------------------------------------------

            if not highres:
                # First, collecting all models
                all_models_for_plotting = all_models_for_plotting[idx_phs_bin]
                all_l68_mods_plotting, all_u68_mods_plotting = all_l68_mods_plotting[idx_phs_bin], all_u68_mods_plotting[idx_phs_bin]

                # ---------------------------------------------------
                #           For transit/occultation
                # ---------------------------------------------------
                
                ## ----------- We can now plot the models (detrended and quantile models)
                ax1.plot(bin_phs, all_models_for_plotting*ppt, color='navy', lw=2.5, zorder=50)
                if quantile_models:
                    ax1.fill_between(bin_phs, y1=all_l68_mods_plotting*ppt, y2=all_u68_mods_plotting*ppt, color='orangered', alpha=0.5, zorder=25)

                # ---------------------------------------------------
                #           For phase curve
                # ---------------------------------------------------
                if phasecurve:
                    ## ----------- We can now plot the models (detrended and quantile models)
                    ax3.plot(bin_phs, all_models_for_plotting, color='navy', lw=2.5, zorder=50)
                    if quantile_models:
                        ax3.fill_between(bin_phs, y1=all_l68_mods_plotting, y2=all_u68_mods_plotting, color='orangered', alpha=0.5, zorder=25)

        if one_plot:
            # ---------------------------------------------------
            #           Figure labels etc.
            # ---------------------------------------------------
            # ------------------------
            #.   Transits
            # ------------------------
            ## First, removing the x-axis labels of top panels
            ax1.xaxis.set_major_formatter(plt.NullFormatter())
            # x-and y-labels
            if not phasecurve:
                if not eclipse:
                    ax1.set_ylabel('Normalised flux')
                else:
                    ax1.set_ylabel('Normalised flux [ppm]')
                ax2.set_ylabel('Residuals [ppm]')
            else:
                ax1.set_ylabel('Normalised flux [ppt]')
                ax2.set_ylabel('Residuals [ppt]')
            
            ax2.set_xlabel('Orbital phase')
            

            # ------------------------
            #.   Phase curves
            # ------------------------
            if phasecurve:
                ## Removing x-axis labels of top panels
                ax3.xaxis.set_major_formatter(plt.NullFormatter())

                ## Now, we will switch the y-axis labels to right side for phase curve panels
                ax3.yaxis.tick_right()
                ax3.tick_params(labelright=True)
                ax3.yaxis.set_label_position('right')
                
                ax4.yaxis.tick_right()
                ax4.tick_params(labelright=True)
                ax4.yaxis.set_label_position('right')

                ## x-and y-labels
                ax3.set_ylabel('Normalised flux [ppm]', rotation=270, labelpad=25)
                
                ax4.set_xlabel('Orbital phase')
                ax4.set_ylabel('Residuals [ppm]', rotation=270, labelpad=25)

        ## Depending on one_plot, we return the appropriate objects
        if not one_plot:
            return figs_all, axs1_all, axs2_all, axs3_all, axs4_all
        else:
            return fig, ax1, ax2, ax3, ax4

    def plot_fake_allan_deviation(self, instruments=None, binmax=10, method='pipe', timeunit=None):
        """Compute and plot per-instrument noise-vs-bin-size curves.

        This is a thin wrapper around ``utils.fake_allan_deviation`` that
        computes residuals for each instrument (data minus the full model)
        and returns the figures, axes and numeric results for further use.

        Parameters
        ----------
        instruments : list or None
            List of instrument names to include. If ``None``, all
            instruments in the juliet dataset (``self.dataset.inames_lc``)
            are processed.
        binmax : int, optional
            Passed to ``utils.fake_allan_deviation`` to control the maximum
            number of bins (default ``10``).
        method : {'pipe','std','rms','astropy'}, optional
            Method used to estimate the scatter of the binned residuals:
            - 'pipe' : use ``pipe_mad`` (median absolute differences estimator)
            - 'std'  : use ``np.nanstd``
            - 'rms'  : use the root-mean-square (``rms`` helper)
            - 'astropy' : use ``astropy.stats.mad_std`` (robust MAD-based std)
        timeunit : str or None, optional
            If provided, forwarded to ``utils.fake_allan_deviation`` to
            force the secondary x-axis time unit; otherwise the helper
            chooses an appropriate unit automatically.

        Returns
        -------
        figs_all : list
            List of matplotlib Figure objects (one per instrument).
        axs_all : list
            List of matplotlib Axes objects corresponding to the figures.
        binsize_all : list
            List of binsize arrays returned for each instrument.
        noise_all : list
            List of noise arrays (ppm) computed for each instrument.
        white_noise_all : list
            List of white-noise expectation arrays (ppm) for each
            instrument.
        """

        # Let's first gather the names of all instruments (we will always do one plot per instrument)
        if instruments != None:
            if len(instruments) != len(self.dataset.inames_lc):
                one_plot = False
        else:
            instruments = self.dataset.inames_lc

        ## All stuff we can save
        figs_all, axs_all = [], []
        binsize_all, noise_all, white_noise_all = [], [], []

        for ins in range(len(instruments)):
            # Getting the data
            tim_ins = self.dataset.times_lc[instruments[ins]][self.idx_time_sort[instruments[ins]]]
            fl_ins = self.dataset.data_lc[instruments[ins]][self.idx_time_sort[instruments[ins]]]

            ## Median model
            full_model = self.models_all_ins[instruments[ins]][1][self.idx_time_sort[instruments[ins]]]

            residuals = fl_ins - full_model

            # And, now, let's plot the Allan deviation
            fig, ax, binsize, noise, white_noise = utils.fake_allan_deviation(times=tim_ins, residuals=residuals, binmax=binmax, method=method, timeunit=timeunit, plot=True)

            figs_all.append(fig)
            axs_all.append(ax)
            binsize_all.append(binsize)
            noise_all.append(noise)
            white_noise_all.append(white_noise)

        return figs_all, axs_all, binsize_all, noise_all, white_noise_all
    
    def plot_corner(self, planet_only=False, save=True):
        """Create a corner plot of selected posterior parameters.


        Parameters
        ----------
        planet_only : bool, optional
            If ``True``, include only planetary parameters in the corner
            plot. If ``False`` (default), include instrumental and noise
            parameters as well when available.
        save : bool, optional
            If ``True`` (default), save the generated figure to
            ``<input_folder>/corner_plot.png``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure produced by ``utils.corner_plot`` containing the
            marginal and pairwise plots for the selected parameters.

        """

        # Access the posteriors
        post_samps = self.res.posteriors['posterior_samples']

        # Arrays for posteriors and labels
        posteriors, labels, titles = [], [], []
        for i in post_samps.keys():
            par = i.split('_')
            if ( 'p1' in par ) or ( 'p2' in par ) or ( 'p3' in par ) or ( 'p4' in par ) or ( 'q1' in par ) or ( 'q2' in par ) or ( 'rho' in par ):
                ## Adding posteriors and labels ( this parameters are added no matter planet_only is True/False)
                if i[0:3] == 'P_p':
                    ## Period parameter: the uncertainty is generally very small, so we can scale it to make it plottable
                    per_exponent_part = np.ceil( np.log10( np.nanstd( post_samps[i] ) ) )
                    per_median = np.nanmedian( post_samps[i] )
                    per = ( post_samps[i] - per_median ) / ( 10**per_exponent_part )

                    ## Saving the posteriors
                    posteriors.append( per )
                    labels.append( i + ' [(P $-$ ' + str(np.around(per_median,5)) + ') / $10^{' + str(int(per_exponent_part)) + '}$ d]' )
                    titles.append( par[0] )
                
                elif i[0:2] == 't0':
                    ## Same for t0, we don't need to write out the whole thing
                    t02 = np.floor( np.nanmedian(post_samps[i]) )

                    posteriors.append( post_samps[i] - t02 )
                    labels.append( i + ' [t0 $-$ ' + str(int(t02)) + ' d]' )
                    titles.append( par[0] )
                
                elif (i[0:3] == 'p_p') or (i[0:4] == 'p1_p') or (i[0:4] == 'p2_p'):
                    ## For Rp/R*, if there are more than 1 instruments, we will label them as 'instrument_name et al.'
                    posteriors.append( post_samps[i] )
                    titles.append( par[0] )
                    if len(i.split('_')) > 3:
                        labels.append('_'.join(i.split('_')[0:3]) + '_et al.')
                    else:
                        labels.append(i)
                
                elif (i[0:2] == 'q1') or (i[0:2] == 'q2'):
                    ## Same for limb-darkening coefficients, if there are more than 1 instruments, we will label them as 'instrument_name et al.'
                    posteriors.append( post_samps[i] )
                    titles.append( par[0] )
                    if len(i.split('_')) > 2:
                        labels.append('_'.join(i.split('_')[0:2]) + '_et al.')
                    else:
                        labels.append(i)
                
                elif ( i[0:2] == 'fp' ) or ( i[0:4] == 'C1_p' ) or ( i[0:2] == 'C2' ) or ( i[0:2] == 'D1' ) or ( i[0:2] == 'D2' ):
                    ## For flux parameters, we will label them as 'instrument_name et al.'
                    posteriors.append( post_samps[i] * 1e6 )
                    titles.append( par[0] )
                    if len(i.split('_')) > 2:
                        labels.append('_'.join(i.split('_')[0:2]) + '_et al. [ppm]')
                    else:
                        labels.append(i + ' [ppm]')

                ## If there are 'sesinomega_p1' and 'secosomega_p1' parameters, we will convert them to 'e_p1' and 'omega_p1'
                elif ( i[0:10] == 'secosomega' ):
                    pl_num = i.split('_')[-1]
                    sesinomega = post_samps['sesinomega_' + pl_num]
                    secosomega = post_samps['secosomega_' + pl_num]

                    e_p = sesinomega**2 + secosomega**2
                    omega_p = np.arctan2( sesinomega, secosomega ) * 180. / np.pi

                    ## First, add sesinomega
                    posteriors.append( post_samps[i] )
                    labels.append( i )
                    titles.append( par[0] )

                    ## And now, e and w
                    posteriors.append( e_p )
                    labels.append( 'e_' + pl_num )
                    titles.append( 'e' )

                    posteriors.append( omega_p )
                    labels.append( r'$\omega$_' + pl_num + ' [deg]' )
                    titles.append( r'$\omega$' )
                
                else:
                    ## Else
                    posteriors.append( post_samps[i] )
                    labels.append( i )
                    titles.append( par[0] )

            else:
                ## Rest of the parameters are only added if planet_only is False
                if not planet_only:
                    if ( i[0:7] != 'unnamed' ) and ( i[0:7] != 'loglike' ):
                        posteriors.append( post_samps[i] )
                        labels.append( i )
                        titles.append( par[0] )
        
        # ------------------------------------------
        #     Now, we can make the corner plot
        # ------------------------------------------

        ## Reducing fontsize specially for this plot
        from matplotlib import rcParams
        rcParams['font.size'] = 10.
        rcParams['axes.labelsize'] = 'medium'
        rcParams['xtick.labelsize'] = 'medium'
        rcParams['ytick.labelsize'] = 'medium'
        rcParams['axes.titlesize'] = 'large'

        fig = utils.corner_plot(samples=posteriors, labels=labels, titles=titles)
        if save:
            plt.savefig(self.input_folder + '/corner_plot.png', dpi=250)

        return fig
    
    def plot_gp(self, instruments=None, highres=False, one_plot=False, figsize=(15/1.5, 6/1.5), pycheops_binning=False, xlabel='GP regressor', robust_errors=False, parameter_vector=None, nos_bin=20):
        """Plot the Gaussian Process component(s) from the fitted `juliet` model.

        This routine computes and plots the GP model (and binned data)
        for one or more instruments present in the current `juliet` dataset.

        Parameters
        ----------
        instruments : list or None
            List of instrument names to plot. If ``None``, all instruments for
            which GP regressors exist in ``self.dataset.GP_lc_arguments`` are
            plotted.
        highres : bool, optional
            If ``True``, compute a high-resolution GP prediction (calls
            ``compute_gp_model``). Default is ``False``.
        one_plot : bool, optional
            If ``True``, combine all instruments into a single figure and
            return a single ``(fig, axs)``. If ``False``, return lists of
            figures and axes (one per instrument). Default ``False``.
        figsize : tuple, optional
            Figure size passed to ``matplotlib``. Default ``(15/1.5, 6/1.5)``.
        pycheops_binning : bool, optional
            Use ``pycheops`` produced binning when set to ``True``.
            Default ``False``.
        xlabel : str, optional
            X-axis label for the GP regressor. Default ``'GP regressor'``.
        robust_errors : bool, optional
            If ``True``, compute GP uncertainties using
            ``compute_gp_model`` instead of using ``juliet`` precomputed errors.
        parameter_vector : array-like or None, optional
            If provided, passed to the GP parameter setter before computing
            predictions (useful for evaluating GP at different parameter
            values).
        nos_bin : int, optional
            Approximate number of bins to use when computing binned data per
            instrument. Default ``20``.

        Returns
        -------
        (fig, axs) or (fig_all, axs_all)
            If ``one_plot`` is ``True``, returns a single tuple ``(fig, axs)``
            for the combined figure. Otherwise returns ``(fig_all, axs_all)``
            where each is a list containing one Figure/Axis per instrument.

        """
        # -------------------------------------------
        #         List of all instruments
        # -------------------------------------------
        ## If instruments=None, we will generate one plot per instruments for all instruments for which GP fitting is done
        if instruments != None:
            ## Let's check if the provided instruments indeed have GP fitting
            for ins in range(len(instruments)):
                if instruments[ins] not in self.dataset.GP_lc_arguments.keys():
                    raise ValueError('GP fitting is not found for instrument ' + instruments[ins])
        else:
            instruments = [i for i in self.dataset.GP_lc_arguments.keys()]
        
        # -------------------------------------------
        #     Computing detrended data and model
        # -------------------------------------------
        self.detrend_data(phmin=0.8, instruments=instruments)
        self.detrend_model(instruments=instruments, phmin=0.8, highres=False)

        # -------------------------------------------
        #      Let's now compute the residuals 
        #        needed to plot the GP model
        #   The total juliet model is:
        #      y = T*F1 + F2 + GP + LM
        #     => GP = y - T*F1 - F2 - LM
        # -------------------------------------------
        gp_model_regressor, self.gp_data = {}, {}
        gp_med_model, gp_up68, gp_lo68 = {}, {}, {}
        for i in range(len(instruments)):
            # We note here that every array should be sorted according to GP regressors (e.g., time or roll angle)
            # We will keep it that way -- everything sorted according GP regressors

            ## -- We also need to compute linear model, if there is any
            if instruments[i] in self.dataset.lm_lc_arguments.keys():
                linear_model = self.models_all_ins[instruments[i]][-1]['lm']
            else:
                linear_model = 0.
            
            # -------------------------------------------
            #   GP data: i.e., data fitted to GP model
            # -------------------------------------------
            self.gp_data[instruments[i]] = self.dataset.data_lc[instruments[i]] - ( self.F1[instruments[i]] * self.planet_only_models[instruments[i]][1] ) -\
                                           self.F2[instruments[i]] - linear_model
            
            # -------------------------------------------
            #  And GP model -- we need to compute them
            # -------------------------------------------
            
            if not highres:
                gp_model_regressor[instruments[i]] = np.copy( self.dataset.GP_lc_arguments[instruments[i]][:, 0] )
                gp_med_model[instruments[i]] = self.all_mods_ins[instruments[i]]['GP']
                if not robust_errors:
                    gp_up68[instruments[i]], gp_lo68[instruments[i]] = self.all_mods_ins[instruments[i]]['GP_uerror'], self.all_mods_ins[instruments[i]]['GP_lerror']
                else:
                    ## In case of robust errors, we need to compute errors from ``compute_gp_model`` function
                    gp_quantiles = self.compute_gp_model(instrument=instruments[i], highres=False, parameter_vector=parameter_vector)
                    gp_up68[instruments[i]], gp_lo68[instruments[i]] = gp_quantiles[1,:], gp_quantiles[2,:]

            else:
                # High-resolution GP regressor per instrument
                predict_time = np.linspace( np.min(self.dataset.GP_lc_arguments[instruments[i]][:, 0]),\
                                            np.max(self.dataset.GP_lc_arguments[instruments[i]][:, 0]), 10000 )
                
                # ------ For highres models, we always need to compute models from ``compute_gp_model`` function
                gp_quantiles = self.compute_gp_model(instrument=instruments[i], highres=True, pred_time=predict_time, parameter_vector=parameter_vector)

                gp_model_regressor[instruments[i]] = predict_time

                gp_med_model[instruments[i]] = gp_quantiles[0,:]
                gp_up68[instruments[i]], gp_lo68[instruments[i]] = gp_quantiles[1,:], gp_quantiles[2,:]
        
        # Now, I think, we have everything -- so, let's plot!
        # -------------------------------------------
        #         Starting figure code:
        #      one plot: do plt.subplots() here
        # -------------------------------------------
        if one_plot:
            fig, axs = plt.subplots(figsize=figsize)
            
            # If there is one plot, then we also need to collect all data to compute binned data
            bin_gp_reg, bin_resids, bin_errors = np.array([]), np.array([]), np.array([])
            
            # Also, in case we don't have highres model, we want to save all models before plotting
            all_regressors_for_plotting, all_models_for_plotting = np.array([]), np.array([])
            all_u68_mods_plotting, all_l68_mods_plotting = np.array([]), np.array([])
        else:
            # We need a list to save all fig and axs objects if we are planning more than one plot
            fig_all, axs_all = [], []

        for i in range(len(instruments)):
            # -------------------------------------------
            #    If not one plot, we need to create
            #    fig and axs object inside for loop
            # -------------------------------------------
            if not one_plot:
                fig, axs = plt.subplots(figsize=figsize)

            # Plotting the "raw" data
            axs.errorbar(self.dataset.GP_lc_arguments[instruments[i]][:,0], self.gp_data[instruments[i]]*1e6, fmt='.', alpha=0.1, color='dodgerblue', zorder=1)
            
            # ------------------------------------------
            #    Now, we can plot the bin data and
            #    quantile models here, if there is
            #  no one plot, else, just save everything
            #    to one big array, to plot outside
            #     the for loop. We can also do
            #        the label thing here.
            # ------------------------------------------
            if not one_plot:
                # First bin-data
                if not pycheops_binning:
                    nbin = int( len( self.gp_data[instruments[i]] ) / nos_bin )
                    bin_tim, bin_fl, bin_fle = juliet.utils.bin_data(x=self.dataset.GP_lc_arguments[instruments[i]][:,0], y=self.gp_data[instruments[i]]*1e6, n_bin=nbin)

                else:
                    binwid = np.ptp( self.dataset.GP_lc_arguments[instruments[i]][:,0] ) / nos_bin
                    bin_tim, bin_fl, bin_fle, _ = utils.lcbin(time=self.dataset.GP_lc_arguments[instruments[i]][:,0], flux=self.gp_data[instruments[i]]*1e6, binwidth=binwid)
                axs.errorbar(bin_tim, bin_fl, yerr=bin_fle, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)
                    
                # And now models
                ## If we don't need highres model, we can simply plot what we have
                ## Else, we need to create highres model first
                axs.plot(gp_model_regressor[instruments[i]], gp_med_model[instruments[i]]*1e6, color='navy', lw=2.5, zorder=50)
                
                # If we are making one plot per instrument (i.e., one_plot=False), we can simply plot quantile models right here
                axs.fill_between(gp_model_regressor[instruments[i]], y1=gp_lo68[instruments[i]]*1e6, y2=gp_up68[instruments[i]]*1e6, color='orangered', alpha=0.5, zorder=25)

                # Labels, limits
                axs.set_xlabel(xlabel)
                axs.set_ylabel('Relative flux [ppm]')

                axs.set_xlim([ np.min(self.dataset.GP_lc_arguments[instruments[i]][:,0]), np.max(self.dataset.GP_lc_arguments[instruments[i]][:,0]) ])
                axs.set_ylim([ np.nanmedian(bin_fl)-5*utils.pipe_mad(bin_fl), np.nanmedian(bin_fl)+5*utils.pipe_mad(bin_fl) ])

                # Saving fig, axs to the list
                fig_all.append(fig)
                axs_all.append(axs)
            else:
                # Saving data to the big list
                bin_gp_reg, bin_resids = np.hstack( (bin_gp_reg, self.dataset.GP_lc_arguments[instruments[i]][:,0]) ), np.hstack( (bin_resids, self.gp_data[instruments[i]]*1e6) )
                bin_errors = np.hstack( (bin_errors, self.detrended_errs[instruments[i]]*1e6) )
                # Saving models to the list
                all_regressors_for_plotting = np.hstack( (all_regressors_for_plotting, gp_model_regressor[instruments[i]]) )
                all_models_for_plotting = np.hstack( (all_models_for_plotting, gp_med_model[instruments[i]]*1e6) )

                ## Let's first save the quantile models
                all_l68_mods_plotting = np.hstack( (all_l68_mods_plotting, gp_lo68[instruments[i]]*1e6) )
                all_u68_mods_plotting = np.hstack( (all_u68_mods_plotting, gp_up68[instruments[i]]*1e6) )
        
        # ------------------------------------------
        #    If one plot, we can now plot the 
        #    bin data, quantile models, labels 
        #    etc. now, outside of the for loop
        # ------------------------------------------
        if one_plot:
            # First, bin data
            if not pycheops_binning:
                nbin = int( len( bin_gp_reg ) / nos_bin / len(instruments) )
                bin_tim, bin_fl, bin_fle = juliet.utils.bin_data(x=bin_gp_reg, y=bin_resids, n_bin=nbin)

            else:
                binwid = np.ptp( bin_gp_reg ) / nos_bin / len(instruments)
                bin_tim, bin_fl, bin_fle, _ = utils.lcbin(time=bin_gp_reg, flux=bin_resids, binwidth=binwid)
            axs.errorbar(bin_tim, bin_fl, yerr=bin_fle, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=100)

            if highres:
                # ------------------------------------------
                #     One plot _and_ highres requires a 
                #          special treatment
                # ------------------------------------------
                ## -- Okay, we need one plot, in high resolution, i.e., we need to call a function to do this.
                idx_gp_reg_sort = np.argsort(bin_gp_reg)
                pred_time = np.linspace(np.min(bin_gp_reg[idx_gp_reg_sort]), np.max(bin_gp_reg[idx_gp_reg_sort]), 10000)
                gp_quantiles = self.compute_gp_model(instrument=None, highres=True, times=bin_gp_reg[idx_gp_reg_sort], pred_time=pred_time,\
                                                     resids=bin_resids[idx_gp_reg_sort]/1e6, errors=bin_errors[idx_gp_reg_sort]/1e6,\
                                                     parameter_vector=parameter_vector)
                
                axs.plot(pred_time, gp_quantiles[0,:]*1e6, color='navy', lw=2.5, zorder=50)
                axs.fill_between(pred_time, y1=gp_quantiles[2,:]*1e6, y2=gp_quantiles[1,:]*1e6, color='orangered', alpha=0.5, zorder=25)

            else:
                ## First, sorting according to regressors, and then plotting the full model
                idx_reg_sort = np.argsort(all_regressors_for_plotting)
                axs.plot(all_regressors_for_plotting[idx_reg_sort], all_models_for_plotting[idx_reg_sort], color='navy', lw=2.5, zorder=50)
                
                # We can plot qunatile models here, if highres=False
                axs.fill_between(all_regressors_for_plotting[idx_reg_sort], y1=all_l68_mods_plotting[idx_reg_sort], y2=all_u68_mods_plotting[idx_reg_sort], color='orangered', alpha=0.5, zorder=25)
            
            # Labels, limits
            axs.set_xlabel(xlabel)
            axs.set_ylabel('Relative flux [ppm]')

            axs.set_xlim([ np.min(bin_gp_reg), np.max(bin_gp_reg) ])
            axs.set_ylim([ np.nanmedian(bin_fl)-5*utils.pipe_mad(bin_fl), np.nanmedian(bin_fl)+5*utils.pipe_mad(bin_fl) ])

        if one_plot:
            return fig, axs
        else:
            return fig_all, axs_all
        
    def compute_gp_model(self, instrument=None, highres=None, times=None, pred_time=None, resids=None, errors=None, parameter_vector=None):
        """This function computes the GP models. It accesses the GP object from the juliet files, and then do GP.compute and GP.predict."""
        
        # If we compute GP model this way, then the errors will always be robust because we will be using gp.predict

        # If time is None: that means that we need model for an instrument (not one model for all instruments)
        #
        # times is not None, means that we need one model for more than one instruments
        # This model is always highres
        # This also assumes that the GP model is the same for all instruments
        
        if times is None:
            # When time is None, means that we haven't provided any data, so we need to extract it from self.
            ## Extracting times (GP regressors) from self.dataset object
            times = self.dataset.GP_lc_arguments[instrument][:,0]
            ## Similarly, extracting GP data and errors from self
            resids, errors = self.gp_data[instrument], self.detrended_errs[instrument]
            
            if highres:
                # Model for one instrument, but still highres model
                pred_time = np.linspace(np.min(times), np.max(times), 10000)
            else:
                # Not highres model
                pred_time = np.copy(times)
        
        else:
            # We don't need any data in this case, since everything is provided, 
            # but we still need a name of an instrument to access GP object
            instrument = [i for i in self.dataset.GP_lc_arguments.keys()][0]

        if parameter_vector is not None:
            self.dataset.lc_options[instrument]['noise_model'].GP.set_parameter_vector(parameter_vector)
        
        # GP.compute
        self.dataset.lc_options[instrument]['noise_model'].GP.compute(t=times, yerr=errors)
        # And GP prediction
        gp_mean, gp_var = self.dataset.lc_options[instrument]['noise_model'].GP.predict(y=resids, t=pred_time, return_var=True)
        gp_std = np.sqrt(gp_var)

        gp_quantiles = np.zeros( (3, len(gp_mean)) )
        gp_quantiles[0, :] = gp_mean
        gp_quantiles[1, :], gp_quantiles[2, :] = gp_mean + gp_std, gp_mean - gp_std

        return gp_quantiles
    
class ApPhoto(object):
    """Helper for aperture photometry on image time-series.

    This class encapsulates common operations required for aperture
    photometry on a data cube of frames (shape ``(N_frames, Ny, Nx)``),
    including centroid estimation, aperture and sky-mask construction,
    simple aperture flux extraction, and pixel-level decorrelation
    (PLD) basis construction.

    Parameters
    ----------
    times : array_like
        1D array of time stamps corresponding to each frame.
    frames : ndarray
        3D data cube with shape ``(N_frames, Ny, Nx)`` containing image
        pixel values for each frame.
    errors : ndarray
        Per-pixel uncertainties with the same shape as ``frames``.
    badpix : ndarray
        2D bad-pixel mask for a single frame (shape ``(Ny, Nx)``).
        A value of zero indicates a bad/ignored pixel; non-zero
        indicates a usable pixel.
    aprad : float or None, optional
        Aperture radius in pixels for circular aperture photometry.
        Used by :meth:`aperture_mask` and :meth:`simple_aperture_photometry`.
        Ignored when ``brightpix`` is True. Default is ``None``.
    sky_rad1 : float or None, optional
        Inner radius (pixels) of the sky annulus. Used by :meth:`sky_mask`
        and :meth:`simple_aperture_photometry` for background estimation.
        If omitted and ``brightpix`` is False, an empty (zero) mask is returned.
        Default is ``None``.
    sky_rad2 : float or None, optional
        Outer radius (pixels) of the sky annulus. Used by :meth:`sky_mask`
        and :meth:`simple_aperture_photometry` for background estimation.
        Default is ``None``.
    brightpix : bool, optional
        If ``True``, select ``nos_brightest`` brightest pixels to form an aperture 
        instead of circular aperture. Used by :meth:`aperture_mask` and :meth:`sky_mask`. 
        Default ``False``.
    nos_brightest : int, optional
        Number of brightest pixels to include in aperture when ``brightpix=True``.
        Default ``12``.
    nos_faintest : int, optional
        Number of faintest pixels to include in background aperture when it is not None and ``brightpix=True``.
        When None, all pixels that are not in aperture are included. Default is ``None``.
    minmax : dict or None, optional
        Optional spatial bounds for brightest pixel selection with keys
        'rmin', 'rmax', 'cmin', 'cmax'. Default is ``None``.

    Attributes
    ----------
    times, frames, errors, badpix : as above
    aprad, sky_rad1, sky_rad2, brightpix, nos_brightest, minmax : as above
    cen_r, cen_c : ndarray
        Per-frame row/column centroid positions (populated by
        :meth:`find_center`).
    ap_bool_mask_pld : ndarray
        2D aperture boolean mask used by PLD routines.
    Psum, Phat, V, eigenvalues, PCA : ndarray
        Intermediate PLD quantities (pixel-sum, normalized fractions,
        PCA eigenvectors/values and projected time-series) populated by
        :meth:`pixel_level_decorrelation`.

    Main methods
    ------------
    identify_crays(clip, niters)
        Flag cosmic-ray affected pixels and update ``badpix``.
    replace_nan(max_iter)
        Iteratively fill NaNs in ``frames`` from neighboring pixels.
    find_center(rmin,rmax,cmin,cmax)
        Compute center-of-flux centroids per frame.
    simple_aperture_photometry(method, plot, ...)
        Extract aperture photometry using manual, photutils, or median methods.
    pixel_level_decorrelation(removeNan)
        Build normalized pixel level light curves and compute PCA basis for PLD.

    Notes
    -----
    The class stores both raw inputs and intermediate results used by
    subsequent calls; many methods modify the instance in-place and
    also return results for convenience.
    """
    def __init__(self, times, frames, errors, badpix, aprad=None, sky_rad1=None, sky_rad2=None, brightpix=False, nos_brightest=12, nos_faintest=None, minmax=None):
        # The data
        self.times = times
        self.frames = frames
        self.errors = errors
        self.badpix = badpix

        # The aperture information
        self.aprad = aprad
        self.sky_rad1, self.sky_rad2 = sky_rad1, sky_rad2
        self.brightpix = brightpix
        self.nos_brightest = nos_brightest
        self.nos_faintest = nos_faintest
        self.minmax = minmax

        if ( self.aprad == None ) and ( not self.brightpix ):
            raise Exception('You need to define an aperture by providing either aperture radius (aprad)\nOr setting brightpix=True, which will form the aperture from bightest pixels.')
    
    def identify_crays(self, clip=5, niters=5):
        """Identify cosmic-ray affected pixels and update the bad-pixel map
           by comparing each data frame with the median frame.

        Parameters
        ----------
        clip : float, optional
            Sigma threshold used to flag cosmic-ray candidates (default
            ``5``).
        niters : int, optional
            Number of iterative passes to update ``self.badpix`` (default
            ``5``).

        Returns
        -------
        badpix : ndarray
            Updated 2D bad-pixel mask (shape ``(Ny, Nx)``) where flagged
            pixels are set to zero and good pixels are one.
        """
        # Masking bad pixels as NaN
        
        for _ in range(niters):
            # Flagging bad data as Nan
            frame_new = np.copy(self.frames)
            frame_new[self.badpix == 0.] = np.nan
            
            # Median frame
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                median_frame = np.nanmedian(frame_new, axis=0)  # 2D frame
                
                # Creating residuals
                resids = frame_new - median_frame[None,:,:]
                
                # Median and std of residuals (both are 2D frames)
                med_resid, std_resid = np.nanmedian(resids, axis=0), np.nanstd(resids, axis=0)
            
            limit = med_resid + (clip*std_resid)
            mask_cr1 = np.abs(resids) < limit[None,:,:]
            self.badpix = mask_cr1*self.badpix

        return self.badpix
    
    def replace_nan(self, max_iter = 50):
        """Fill NaN entries in the data array by the mean of neighbouring pixels.

        On each iteration the array is rolled along each axis to build a
        collection of nearest-neighbour shifts; the mean of these shifted
        arrays is used to replace NaNs. Iteration stops when there are no
        remaining NaNs or when ``max_iter`` is reached.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of fill iterations to perform (default ``50``).

        Returns
        -------
        frames : ndarray
            The data array with NaNs replaced (also modified
            in-place and returned).
        """
        shape = np.append([2*self.frames.ndim], self.frames.shape)
        interp_cube = np.zeros(shape)
        axis = tuple(range(self.frames.ndim))
        shift0 = np.zeros(self.frames.ndim, int)
        shift0[0] = 1
        shift = []     # Shift list will be [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for n in range(self.frames.ndim):
            shift.append(tuple(np.roll(-shift0, n)))
            shift.append(tuple(np.roll(shift0, n)))
        
        for _ in range(max_iter):
            for n in range(2*self.frames.ndim):
                interp_cube[n] = np.roll(self.frames, shift[n], axis = axis)   # interp_cube would be (4, data.shape[0], data.shape[1]) sized array
            with warnings.catch_warnings():                                 # with shifted position in each element (so that we can take its mean)
                warnings.simplefilter('ignore', RuntimeWarning)
                mean_data = np.nanmean(interp_cube, axis=0)
            self.frames[np.isnan(self.frames)] = mean_data[np.isnan(self.frames)]
            if np.sum(np.isnan(self.frames)) == 0:
                break
        
        return self.frames
    
    def plot_median_image(self):
        """This function plots the median image frame
        
        Returns
        -------
        fig : matplotlib.figure.Figure or None
            Figure of median image.
        im : matplotlib.image.AxesImage or None
            Image object from the median image plot.
        axs : ndarray or None
            Axes of the figure.
        """
        fig, axs = plt.subplots( figsize=(5,5) )
        im = axs.imshow( np.nanmedian(self.frames, axis=0), interpolation='none' )
        axs.set_title('Median image')

        return fig, im, axs
    
    def find_center(self, rmin=None, rmax=None, cmin=None, cmax=None, plot=False):
        """Compute center-of-flux centroids for each frame in the 3D data cube.

        The centroid for each frame is computed as the center of flux
        (i.e. sum(x*I)/sum(I)) over either the full image or an optional
        subimage defined by the row/column limits ``rmin,rmax,cmin,cmax``.
        Results are stored on the instance as ``self.cen_r`` and
        ``self.cen_c`` and also returned.

        Parameters
        ----------
        rmin, rmax, cmin, cmax : int or None, optional
            Optional integer bounds (rows and columns) defining a subimage
            to use for the centroid calculation. If omitted, the full
            frame is used.
        plot : bool, optional
            If ``True``, produce a plot of centroids as a function of
            time (default ``False``).

        Returns
        -------
        cen_r, cen_c : ndarray
            1D arrays (length ``N_frames``) containing the row and column
            centroid positions (in pixel coordinates) for each frame.
        fig : matplotlib.figure.Figure or None
            Figure of centroids vs time, if ``plot=True``, otherwise ``None``.
        axs : ndarray or None
            Axes of the figure if produced, otherwise ``None``.
        """
        self.cen_r, self.cen_c = np.zeros(self.frames.shape[0]), np.zeros(self.frames.shape[0])

        for i in range(len(self.cen_r)):
            # Row is the first index, column is the second index
            # First find the subimage if min & max row, cols are provided
            if (rmin != None)&(rmax != None)&(cmin == None)&(cmax == None):
                subimg = self.frames[i, rmin:rmax, :]
            elif (rmin == None)&(rmax == None)&(cmin != None)&(cmax != None):
                subimg = self.frames[i, :, cmin:cmax]
            elif (rmin != None)&(rmax != None)&(cmin != None)&(cmax != None):
                subimg = self.frames[i, rmin:rmax, cmin:cmax]
            else:
                subimg = np.copy(self.frames[i,:,:])
            
            # Row and column indices
            row_idx, col_idx = np.arange(subimg.shape[0]), np.arange(subimg.shape[1])

            # And now the center of *subimage*
            cen_r_sub = np.nansum(row_idx * np.nansum(subimg, axis=1)) / np.nansum(subimg.flatten())
            cen_c_sub = np.nansum(col_idx * np.nansum(subimg, axis=0)) / np.nansum(subimg.flatten())

            # This was the center of subimage, let's transform that to image coordinates
            if (rmin != None)&(cmin == None):
                self.cen_r[i], self.cen_c[i] = cen_r_sub + rmin, cen_c_sub
            elif (rmin == None)&(cmin != None):
                self.cen_r[i], self.cen_c[i] = cen_r_sub, cen_c_sub + cmin
            elif (rmin != None)&(cmin != None):
                self.cen_r[i], self.cen_c[i] = cen_r_sub + rmin, cen_c_sub + cmin
            else:
                self.cen_r[i], self.cen_c[i] = cen_r_sub, cen_c_sub
        
        if plot:
            fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(15/2,10/2), sharex=True)

            axs[0].plot(self.times - self.times[0], self.cen_r, 'k-')
            axs[0].set_ylabel('Row')

            axs[1].plot(self.times - self.times[0], self.cen_c, 'k-')
            axs[1].set_xlabel('Time (BJD)')
            axs[1].set_ylabel('Column')

            axs[1].set_xlim([ 0., np.max( self.times - self.times[0] ) ])
        else:
            fig, axs = None, None

        return self.cen_r, self.cen_c, fig, axs
    
    def aperture_mask(self, plot=False):
        """Build a binary aperture mask (and masked frame/error arrays).

        By default a circular aperture is created on the median image using
        the median centroid position computed from ``self.cen_r`` and
        ``self.cen_c``. Alternatively, when ``self.brightpix=True``, the mask is
        formed by selecting the ``self.nos_brightest`` brightest pixels from the
        median image (optionally limited by ``self.minmax`` bounds).

        Parameters
        ----------
        plot : bool, optional
            If ``True``, produce a plot of median image with aperture
            plotted on the top (default ``False``).

        Returns
        -------
        aper_fl_mask : ndarray
            ``self.frames`` multiplied by the 2D aperture boolean mask
            (shape ``(N_frames, Ny, Nx)``).
        aper_err_mask : ndarray
            ``self.errors`` multiplied by the 2D aperture boolean mask.
        ap_bool_msk : ndarray
            2D binary (0/1) mask describing the aperture on a single frame
            (shape ``(Ny, Nx)``).
        fig : matplotlib.figure.Figure or None
            Figure of median image and aperture mask, if ``plot=True``, otherwise ``None``.
        im : matplotlib.image.AxesImage or None
            Image object from the median image plot if ``plot=True``, otherwise ``None``.
        axs : ndarray or None
            Axes of the figure if produced, otherwise ``None``.

        Notes
        -----
        Uses instance attributes ``self.aprad``, ``self.brightpix``,
        ``self.nos_brightest``, and ``self.minmax`` to control aperture
        construction.
        """

        # Median image
        med_img = np.nanmedian(self.frames, axis=0)
        # Now, creating mask
        ap_bool_msk = np.zeros(med_img.shape)
        
        if not self.brightpix:
            # Let's first find a distance array which will contain the distance of each pixels from the center
            idx_arr_r, idx_arr_c = np.meshgrid(np.arange(med_img.shape[0]), np.arange(med_img.shape[1]))
            idx_arr_r, idx_arr_c = np.transpose(idx_arr_r), np.transpose(idx_arr_c)
            
            # Both of above array would be of the dimension of the image array, and we can get row and column index by doing,
            # idx_arr_r[row, col] = row index and idx_arr_c[row, col] = col index

            # Distance array would give distace of each pixel from the center
            distance = np.sqrt(((idx_arr_r - np.nanmedian(self.cen_r))**2) + ((idx_arr_c - np.nanmedian(self.cen_c))**2))

            ap_bool_msk[distance < self.aprad] = 1.

        else:
            # Sorting all pixels values; so that we can select first N bright pixels in the aperture
            med_img_flat = med_img.flatten()
            med_img_sorted = np.sort(med_img_flat)

            # minmax != None
            if self.minmax is not None:
                rmin, rmax = self.minmax['rmin'], self.minmax['rmax']
                cmin, cmax = self.minmax['cmin'], self.minmax['cmax']
            else:
                rmin, rmax = 0, 10000
                cmin, cmax = 0, 10000

            tot_num = True
            ap_pix_nos, i = 0, 0
            while tot_num:
                if np.isnan(med_img_sorted[-1-i]):
                    i = i + 1
                    continue
                else:
                    ind = np.where(med_img == med_img_sorted[-1-i])
                    # To prevent aperture being far from the center
                    if (ind[0][0] > rmax) or (ind[0][0] < rmin) or (ind[1][0] < cmin) or (ind[1][0] > cmax):
                        i = i + 1
                        continue
                    else:
                        ap_bool_msk[ind[0][0], ind[1][0]] = 1.

                        ap_pix_nos = ap_pix_nos + 1
                        if ap_pix_nos>self.nos_brightest:
                            tot_num = False
                        i = i + 1

        # And, aperture mask
        aper_fl_mask, aper_err_mask = self.frames * ap_bool_msk[None, :, :], self.errors * ap_bool_msk[None, :, :]

        if plot:
            # Figure code
            fig, axs = plt.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True)
            im = axs[0].imshow(med_img, interpolation='none')
            axs[0].imshow(ap_bool_msk, alpha=0.5)

            axs[1].imshow(ap_bool_msk, interpolation='none')

            axs[0].set_title('Median image + Aperture mask')
            axs[1].set_title('Aperture mask')
        else:
            fig, im, axs = None, None, None


        return aper_fl_mask, aper_err_mask, ap_bool_msk, fig, im, axs
    
    
    def sky_mask(self, plot=False):
        """Create a sky-annulus mask (and masked frame/error arrays).

        When ``self.brightpix`` is False this builds a circular annulus defined by
        ``self.sky_rad1`` (inner radius) and ``self.sky_rad2`` (outer radius) around 
        the median centroid position. When ``self.brightpix`` is True the function instead
        selects the ``self.nos_brightest`` brightest pixels (optionally limited
        by ``self.minmax``) to define the aperture, and rest of the pixels as background.

        Parameters
        ----------
        plot : bool, optional
            If ``True``, produce a plot of median image with sky mask
            plotted on the top (default ``False``).

        Returns
        -------
        sky_fl_mask : ndarray
            ``self.frames`` multiplied by the 2D sky boolean mask
            (shape ``(N_frames, Ny, Nx)``).
        sky_err_mask : ndarray
            ``self.errors`` multiplied by the 2D sky boolean mask.
        sky_bool_msk : ndarray
            2D binary (0/1) mask describing the sky annulus on a single
            frame (shape ``(Ny, Nx)``).
        fig : matplotlib.figure.Figure or None
            Figure of median image and sky mask, if ``plot=True``, otherwise ``None``.
        im : matplotlib.image.AxesImage or None
            Image object from the median image plot if ``plot=True``, otherwise ``None``.
        axs : ndarray or None
            Axes of the figure if produced, otherwise ``None``.

        Notes
        -----
        Uses instance attributes ``self.sky_rad1``, ``self.sky_rad2``,
        ``self.brightpix``, ``self.nos_brightest``, and ``self.minmax`` to
        control sky mask construction.
        """

        # Median image
        med_img = np.nanmedian(self.frames, axis=0)

        # Now, creating mask
        sky_bool_msk = np.zeros(med_img.shape)

        if ( self.sky_rad1 == None ) and ( self.sky_rad2 == None ) and ( not self.brightpix ):
            pass

        else:
            if ( not self.brightpix ) or ( ( self.sky_rad1 != None ) and ( self.sky_rad2 != None ) ):
                # Let's first find a distance array which will contain the distance of each pixels from the center
                idx_arr_r, idx_arr_c = np.meshgrid(np.arange(med_img.shape[0]), np.arange(med_img.shape[1]))
                idx_arr_r, idx_arr_c = np.transpose(idx_arr_r), np.transpose(idx_arr_c)
                
                # Both of above array would be of the dimension of the image array, i.e, we can get the row and column index by doing.
                # idx_arr_r[row, col] = row index and idx_arr_c[row, col] = col index

                # Distance array would give distace of each pixel from the center
                distance = np.sqrt(((idx_arr_r - np.nanmedian(self.cen_r))**2) + ((idx_arr_c - np.nanmedian(self.cen_c))**2))

                sky_bool_msk[(distance > self.sky_rad1)&(distance < self.sky_rad2)] = 1.
            
            else:
                # Sorting all pixels values; so that we can select first N bright pixels in the aperture
                med_img_flat = med_img.flatten()
                med_img_sorted = np.sort(med_img_flat)

                # minmax != None
                if self.minmax is not None:
                    rmin, rmax = self.minmax['rmin'], self.minmax['rmax']
                    cmin, cmax = self.minmax['cmin'], self.minmax['cmax']
                else:
                    rmin, rmax = 0, 10000
                    cmin, cmax = 0, 10000

                if self.nos_faintest == None:
                    sky_bool_msk = np.ones(med_img.shape)
                    tot_num = True
                    ap_pix_nos, i = 0, 0
                    while tot_num:
                        if np.isnan(med_img_sorted[-1-i]):
                            i = i + 1
                            continue
                        else:
                            ind = np.where(med_img == med_img_sorted[-1-i])
                            # To prevent aperture being far from the center
                            if (ind[0][0] > rmax) or (ind[0][0] < rmin) or (ind[1][0] < cmin) or (ind[1][0] > cmax):
                                i = i + 1
                                continue
                            else:
                                sky_bool_msk[ind[0][0], ind[1][0]] = 0.

                                ap_pix_nos = ap_pix_nos + 1
                                if ap_pix_nos>self.nos_brightest:
                                    tot_num = False
                                i = i + 1
                else:
                    sky_bool_msk = np.zeros(med_img.shape)
                    tot_num = True
                    ap_pix_nos, i = 0, 0
                    while tot_num:
                        if np.isnan(med_img_sorted[i]):
                            i = i + 1
                            continue
                        else:
                            ind = np.where(med_img == med_img_sorted[i])
                            # To prevent aperture being far from the center
                            if (ind[0][0] < rmax) and (ind[0][0] > rmin) and (ind[1][0] > cmin) and (ind[1][0] < cmax):
                                i = i + 1
                                continue
                            else:
                                sky_bool_msk[ind[0][0], ind[1][0]] = 1.

                                ap_pix_nos = ap_pix_nos + 1
                                if ap_pix_nos>self.nos_faintest:
                                    tot_num = False
                                i = i + 1

        # And, sky mask
        sky_fl_mask, sky_err_mask = self.frames * sky_bool_msk[None, :, :], self.errors * sky_bool_msk[None, :, :]

        if plot:
            # Figure code
            fig, axs = plt.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True)
            im = axs[0].imshow(med_img, interpolation='none')
            axs[0].imshow(sky_bool_msk, alpha=0.5)

            axs[1].imshow(sky_bool_msk, interpolation='none')

            axs[0].set_title('Median image + Bkg mask')
            axs[1].set_title('Bkg mask')
        else:
            fig, im, axs = None, None, None

        return sky_fl_mask, sky_err_mask, sky_bool_msk, fig, im, axs
    
    def simple_aperture_photometry(self, method='manual', plot=False, bkg_corr='mean', robust_err=False, **kwargs):
        """Compute aperture photometry for each frame in the data cube.

        The function supports three methods:
        - ``manual``: sums pixels inside a binary aperture mask and subtracts
          the estimated sky background per pixel computed from a sky annulus.
        - ``photutils``: uses ``photutils`` aperture classes to compute
          aperture sums and background statistics per frame (requires
          ``photutils`` to be installed). Additional keyword arguments are
          forwarded to the photutils helpers.
        - ``median``: returns the per-frame median of the aperture pixels
          (useful as a robust estimator or for quick diagnostics).

        Parameters
        ----------
        method : {'manual','photutils','median'}, optional
            Photometry method to use. Default is ``'manual'``.
        plot : bool, optional
            If ``True``, produce a plot of aperture photometry, flux vs time (default ``False``).
        bkg_corr : str, optional
            The method to compute sky background flux, can be either mean or median. Default is ``mean``.
        robust_err : bool, optional
            Computes robust errors on aperture photometry by including scatter in sky background flux.
            Default is ``False``.
        **kwargs : dict
            Additional keyword arguments forwarded to ``photutils``
            routines when ``method='photutils'`` (for example, custom
            aperture or statistic options).

        Returns
        -------
        ap_flux : ndarray
            1D array (length N_frames) with the background-subtracted
            aperture flux for each frame (same units as ``frames``).
        ap_flux_err : ndarray
            1D array (length N_frames) with the estimated uncertainty on
            the aperture flux for each frame.
        tot_sky_bkg : ndarray
            1D array (length N_frames) with the total sky background
            contribution (summed over aperture pixels) subtracted from
            each frame. For methods that do not estimate a sky background
            this will be zeros.
        fig : matplotlib.figure.Figure or None
            Figure of aperture photometry vs time, if ``plot=True``, otherwise ``None``.
        axs : ndarray or None
            Axes of the figure if produced, otherwise ``None``.

        Notes
        -----
        - The function expects centroids to be available as
          ``self.cen_r`` and ``self.cen_c`` (one per frame) before being
          called.
        - Uses instance attributes ``self.aprad``, ``self.sky_rad1``,
          ``self.sky_rad2``, ``self.brightpix``, ``self.nos_brightest``,
          and ``self.minmax`` to control aperture and background extraction.
        - When using ``photutils``, this routine constructs
          ``CircularAperture``/``CircularAnnulus`` objects per frame and
          uses ``ApertureStats`` and ``aperture_photometry``; errors are
          estimated combining aperture sum errors and annulus statistics
          following standard propagation.
        """
        
        if method == 'manual':
            # Let's first obtain sky background flux per pixel (If sky radii are not None)
            sky_fl_mask, sky_err_mask, sky_bool_mask, _, _, _ = self.sky_mask()
            # Now, the aperture flux
            aper_fl_mask, aper_err_mask, ap_bool_mask, _, _, _ = self.aperture_mask()

            ## Expression of Sky flux per pixel and sky error per pixel
            if np.nansum(sky_bool_mask) != 0:
                # --------------------- First, calculate the mean sky flux per pix
                if bkg_corr == 'mean':
                    sky_flx_per_pix = np.nansum(sky_fl_mask, axis=(1,2)) / np.nansum(sky_bool_mask)
                elif bkg_corr == 'median':
                    # Replacing 0s with Nan, so that they don't interfere with median computation
                    sky_fl_mask12 = np.copy( sky_fl_mask )
                    sky_fl_mask12[ sky_fl_mask12 == 0. ] = np.nan

                    sky_flx_per_pix = np.nanmedian( sky_fl_mask12, axis=(1,2) )
                else:
                    raise Exception('The background can be computed either using mean or median.')
                
                # --------------------- And, now, errors on the mean sky flux per pix
                sky_flx_err_per_pix = np.sqrt( np.nansum( sky_err_mask**2, axis=(1,2) ) ) / np.nansum(sky_bool_mask)
                if robust_err:
                    # Replacing 0s with Nan, so that they don't interfere with std computation
                    sky_fl_mask12 = np.copy( sky_fl_mask )
                    sky_fl_mask12[ sky_fl_mask12 == 0. ] = np.nan
                    
                    sky_flx_err_per_pix = np.sqrt( sky_flx_err_per_pix**2 + ( np.nanstd( sky_fl_mask12, axis=(1,2) ) )**2 )
            else:
                sky_flx_per_pix, sky_flx_err_per_pix = 0., 0.

            tot_sky_bkg = np.nansum(ap_bool_mask)*sky_flx_per_pix
            ap_flux = np.nansum(aper_fl_mask, axis=(1,2)) - tot_sky_bkg
            ap_flux_err = np.sqrt( np.nansum( aper_err_mask**2, axis=(1,2) ) + ( (np.nansum(ap_bool_mask) * sky_flx_err_per_pix)**2 ) )

        elif method == 'photutils':

            ap_flux, ap_flux_err, tot_sky_bkg = np.zeros( self.frames.shape[0] ), np.zeros( self.frames.shape[0] ), np.zeros( self.frames.shape[0] )

            for t in range(len(ap_flux)):
                # First let's perform the background subtraction
                if (self.sky_rad1 == None) and (self.sky_rad2 == None):
                    bkg_mean, bkg_std = 0., 0.
                else:
                    sky_aper = CircularAnnulus((int(self.cen_c[t]), int(self.cen_r[t])), r_in=self.sky_rad1, r_out=self.sky_rad2)
                    sky_aperstats = ApertureStats(self.frames[t,:,:], sky_aper, **kwargs)
                    if bkg_corr == 'mean':
                        bkg_mean, bkg_std = sky_aperstats.mean, sky_aperstats.std   # Mean sky background per pixel
                    elif bkg_corr == 'median':
                        bkg_mean, bkg_std = sky_aperstats.median, sky_aperstats.std   # Mean sky background per pixel
                    else:
                        raise Exception('The background can be computed either using mean or median.')
                
                # Now computing the aperture flux
                circ_aper = CircularAperture((self.cen_c[t], self.cen_r[t]), r=self.aprad)
                ap_phot = aphot(data=self.frames[t,:,:], apertures=circ_aper, error=self.errors[t,:,:], **kwargs)
                total_sky_bkg = bkg_mean * circ_aper.area_overlap(self.frames[t,:,:], **kwargs)
                phot_bkgsub = ap_phot['aperture_sum'] - total_sky_bkg  # Background subtraction

                ## Error estimation in background subtracted photometry
                phot_bkgsub_err = np.sqrt( (ap_phot['aperture_sum_err']**2) + ( (circ_aper.area_overlap(self.frames[t,:,:], **kwargs) * bkg_std)**2 ) )

                ## Results
                ap_flux[t], ap_flux_err[t] = phot_bkgsub[0], phot_bkgsub_err[0]
                tot_sky_bkg[t] = total_sky_bkg

        elif method == 'median':
            aper_fl_mask, aper_err_mask, ap_bool_mask, _, _, _ = self.aperture_mask()
            tot_sky_bkg = np.zeros(aper_fl_mask.shape[0])
            aper_fl_mask[aper_fl_mask == 0.] = np.nan
            ap_flux, ap_flux_err = np.nanmedian(aper_fl_mask, axis=(1,2)), np.sqrt( np.nanmedian( aper_fl_mask, axis=(1,2) ) )
        
        else:
            raise Exception('Please enter correct method...\nMethod can either be manual, photutils, or median.')
        
        if plot:
            fig, axs = plt.subplots()
            axs.errorbar(self.times - self.times[0], ap_flux/np.nanmedian(ap_flux), yerr=ap_flux_err/np.nanmedian(ap_flux), fmt='.', color='dodgerblue')
            
            axs.set_xlim( [0., np.max(self.times - self.times[0]) ] )
            axs.set_xlabel( 'Time [BJD - {:.2f}]'.format(self.times[0]) )
            axs.set_ylabel('Relative flux')
        else:
            fig, axs = None, None
            
        return ap_flux, ap_flux_err, tot_sky_bkg, fig, axs
    
    def growth_function(self, rmin, rmax, noise='pipe', plot=False, **kwargs):
        """Compute the growth curve (flux vs aperture radius) and noise.

        Parameters
        ----------
        rmin : int
            Minimum aperture radius (inclusive) in pixels.
        rmax : int
            Maximum aperture radius (exclusive) in pixels.
        noise : {'pipe','std','rms','astropy'}, optional
            Method used to estimate scatter for each aperture radius.
            Default is 'pipe' (uses ``utils.pipe_mad``).
        plot : bool, optional
            If ``True``, produce a plot of the growth function (default ``False``).
        **kwargs : dict
            Additional keyword arguments forwarded to
            ``simple_aperture_photometry`` (for example sky annulus
            parameters).

        Returns
        -------
        ap_flux_radii : ndarray
            Median aperture flux for each radius in the tested range.
        noise_radii : ndarray
            Estimated noise metric (in ppm) for each radius.
        fig : matplotlib.figure.Figure or None
            Figure of the growth function, if ``plot=True``, otherwise ``None``.
        axs : ndarray or None
            Axes of the figure if produced, otherwise ``None``.
        """
        radii = np.arange(rmin, rmax, 1)
        ap_flux_radii, noise_radii = np.zeros(len(radii)), np.zeros(len(radii))

        # Choosing the method to compute noise
        if noise == 'pipe':
            noise_func = utils.pipe_mad
        elif noise == 'std':
            noise_func = np.nanstd
        elif noise == 'rms':
            noise_func = utils.rms
        elif noise == 'astropy':
            noise_func = lambda x: mad_std(x, ignore_nan=True)
        else:
            raise ValueError("Method should be one of 'pipe', 'std', 'rms', or 'astropy'.")
        
        for r in tqdm(range(len(radii))):
            if not self.brightpix:
                self.aprad = radii[r]
            else:
                self.nos_brightest = radii[r]
            ap_fl, _, _, _, _ = self.simple_aperture_photometry(**kwargs)
            ap_flux_radii[r] = np.nanmedian( ap_fl )
            noise_radii[r] = noise_func( ap_fl / np.nanmedian( ap_fl ) ) * 1e6

        if plot:
            fig, axs1 = plt.subplots()

            color1 = 'orangered'
            axs1.plot(radii, ap_flux_radii, color=color1)
            axs1.set_xlabel('Radius')
            axs1.set_ylabel('Growth function', color=color1)
            axs1.tick_params(axis='y', which='both', color=color1,  labelcolor=color1)
            
            color2 = 'royalblue'
            axs2 = axs1.twinx()
            axs2.plot(radii, noise_radii, color=color2)
            axs2.axvline(radii[np.argmin(noise_radii)], color='dimgrey', ls='--', lw=1.)
            axs2.set_ylabel('Median Absolute Deviation [ppm]', color=color2, rotation=270, labelpad=25)
            
            axs2.tick_params(axis='y', which='both', color=color2, labelcolor=color2)
            axs2.spines['right'].set_color(color2)
            axs2.spines['left'].set_color(color1)

            axs1.set_xlim([ np.min(radii), np.max(radii) ])

            fig.tight_layout()
        else:
            fig, axs1, axs2 = None, None, None

        return ap_flux_radii, noise_radii, fig, [axs1, axs2]
    
    def pixel_level_decorrelation(self, removeNan=False):
        """Prepare pixel-level decorrelation (PLD) basis functions.

        The method builds the normalized pixel-level light curves (Phat) inside
        the chosen aperture and performs a PCA (principal componene analysis) 
        on them to produce a set of orthogonal basis vectors (``self.PCA``) with 
        singular values and eigenvectors stored on the instance.

        Parameters
        ----------
        removeNan : bool, optional
            If ``True``, remove frames that contain NaNs in the basis
            before performing PCA.

        Returns
        -------
        V : ndarray
            Matrix of eigenvectors from the PCA (shape: n_components, n_pixels).
        eigenvalues : ndarray
            Eigenvalues corresponding to each PCA component.
        PCA : ndarray
            Projected PCA time-series (shape: n_components, n_frames).
        """
        _, _, self.ap_bool_mask_pld, _, _, _ = self.aperture_mask()
        sky_fl, _, sky_bool, _, _, _ = self.sky_mask()

        if np.nansum(sky_bool) == 0.:
            sky_bkg_per_pix = 0.
        else:
            sky_bkg_per_pix = np.nansum(sky_fl, axis=(1,2)) / np.nansum(sky_bool)

        self.idxr, self.idxc = np.where(self.ap_bool_mask_pld == 1)

        pixel_fluxes = np.zeros( ( self.frames.shape[0], int(np.sum(self.ap_bool_mask_pld)) ) )
        for r in range(len(self.idxr)):
            pixel_fluxes[:,r] = self.frames[:, int(self.idxr[r]), int(self.idxc[r])] - sky_bkg_per_pix
        
        # Calculating Phat
        self.Psum = np.nansum( pixel_fluxes, axis=1 )
        self.Phat = pixel_fluxes / self.Psum[:, None]

        if removeNan:
            id0, id1 = np.where(np.isnan(self.Phat))
            idx_nan = np.ones(self.Phat.shape[0], dtype=bool)
            idx_nan[id0] = False
            self.Phat = self.Phat[idx_nan, :]
            self.times = self.times[idx_nan]
        else:
            idx_nan = np.ones(self.Phat.shape[0], dtype=bool)

        self.V, self.eigenvalues, self.PCA = utils.classic_PCA(self.Phat.T)

        return self.V, self.eigenvalues, self.PCA
    
    def plot_correlation_matrices(self):
        """Plot correlation matrices before and after PCA on pixel-level light curves.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the two correlation matrix subplots.
        axs : ndarray
            Array of two Axes objects for the before/after PCA plots.
        im_list : list
            List with the two AxesImage objects (useful for colorbars).
        """
        # Calculating PCA correlation matrix
        CorrelationMatrix = np.abs(np.corrcoef(self.Phat.T))
        PCACorrelationMatrix = np.abs(np.corrcoef(self.PCA))

        from matplotlib import rcParams
        rcParams['xtick.direction'] = 'out'
        rcParams['ytick.direction'] = 'out'
        
        # Actual figure code
        fig, axs = plt.subplots(1, 2, figsize=(10,5), sharex=True, sharey=True)
        
        im1 = axs[0].imshow(CorrelationMatrix, cmap='magma', vmin=0, vmax=1)
        im2 = axs[1].imshow(PCACorrelationMatrix, cmap='magma', vmin=0, vmax=1)
        
        axs[0].set_xlabel('Element $i$')
        axs[1].set_xlabel('Element $i$')
        
        axs[0].set_ylabel('Element $j$')

        axs[0].set_title('Before PCA')
        axs[1].set_title('After PCA')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.1, 0.015, 0.75])
        fig.colorbar(im2, cax=cbar_ax)

        rcParams['xtick.direction'] = 'in'
        rcParams['ytick.direction'] = 'in'

        return fig, axs, [im1, im2]
    
    def plot_eigenvectors(self):
        """Visualize the first 10 PCA eigenvectors as spatial maps.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the eigenvector images.
        axs : ndarray
            Array of Axes for each eigenvector subplot.
        """
        comps = 0

        fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(15, 5), sharex=True, sharey=True)
        for r in range(axs.shape[0]):
            for c in range(axs.shape[1]):
                eg_vec = np.zeros(self.ap_bool_mask_pld.shape)
                for k in range(len(self.idxr)):
                    eg_vec[self.idxr[k], self.idxc[k]] = self.V[comps, k]
                im = axs[r,c].imshow(eg_vec, interpolation='none', cmap='plasma')
                im.set_clim([-0.22, 0.22])
                axs[r,c].set_xlim([np.min(self.idxc)-3, np.max(self.idxc)+3])
                axs[r,c].set_ylim([np.min(self.idxr)-3, np.max(self.idxr)+3])
                axs[r,c].text(np.max(self.idxc)+1.5, np.max(self.idxr)+2, comps+1, fontweight='bold')#, backgroundcolor='white')
                comps = comps + 1
        fig.suptitle('First 10 eigenvectors from PCA analysis', fontsize=16)

        return fig, axs
    
    def plot_pcs(self, nmax=10):
        """Plot the PC time-series (principal components) up to `nmax`.

        Parameters
        ----------
        nmax : int, optional
            Number of principal components to plot (default 10).

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the stacked PC plots.
        axs : ndarray
            Array of Axes objects, one per plotted PC.
        """
        fig, axs = plt.subplots(nrows=int(nmax), ncols=1, figsize=(15,10), sharex=True)
        for i in range(int(nmax)):
            axs[i].plot(self.times - self.times[0], self.PCA[i,:], 'k-')
            axs[i].set_ylabel('PC ' + str(i+1))
            med, std = np.nanmedian(self.PCA[i,:]), mad_std(self.PCA[i,:])
            axs[i].set_ylim([med-4*std, med+4*std])
        plt.xlabel('Time (BJD - {:.4f})'.format(self.times[0]))

        return fig, axs
    
    def pld_correction(self, pc_max=5, plot=False):
        """Apply PLD correction using the first `pc_max` principal components.

        Parameters
        ----------
        pc_max : int, optional
            Number of PCA components to include in the PLD fit (default 5).
        plot : bool, optional
            If ``True``, produce a diagnostic plot comparing the raw and
            PLD-predicted flux (default ``False``).

        Returns
        -------
        Psum : ndarray
            Sum of pixel fluxes (SAP flux) before correction.
        prediction : ndarray
            PLD model prediction for the flux (same length as ``Psum``).
        fig : matplotlib.figure.Figure or None
            Diagnostic figure if ``plot=True``, otherwise ``None``.
        axs : ndarray or None
            Axes of the diagnostic figure if produced, otherwise ``None``.
        """
        X = np.vstack(( np.ones(len(self.Psum)), self.PCA[0:pc_max,:] ))
        
        # Fit:
        result = np.linalg.lstsq(X.T, self.Psum, rcond=None)
        coeffs = result[0]
        prediction = np.dot(coeffs, X)

        print('>>> --- MAD of uncorrected light curve is: {:.2f} ppm'.format(utils.pipe_mad(self.Psum/np.median(self.Psum))*1e6))
        print('>>> --- MAD of PLD corrected light curve is: {:.2f} ppm'.format(utils.pipe_mad(self.Psum/prediction)*1e6))

        if plot:
            fig, axs = plt.subplots(2, 1, figsize=(15/1.5,10/1.5), sharex=True, sharey=True)

            axs[0].errorbar(self.times-self.times[0], self.Psum/np.median(self.Psum), fmt='.', c='orangered', label='SAP Flux')
            axs[0].plot(self.times-self.times[0], prediction/np.median(self.Psum), c='darkgreen', label='PLD prediction',alpha=0.7, zorder=10)
            axs[0].set_ylabel('Relative Flux')
            axs[0].legend(loc='best')
            axs[0].grid()

            axs[1].errorbar(self.times-self.times[0], self.Psum/prediction, fmt='.', c='cornflowerblue')
            axs[1].set_xlabel('Time [BJD - {:.2f}]'.format(self.times[0]))
            axs[1].set_ylabel('Relative Flux')
            axs[1].grid()

            axs[1].set_xlim([ 0., np.max(self.times-self.times[0]) ])

        else:
            fig, axs = None, None

        return self.Psum, prediction, fig, axs
    
    def noise_with_pcs(self, pc_max, noise='pipe', plot=False):
        """Estimate noise (MAD or other) as a function of included PCs.

        Parameters
        ----------
        pc_max : int
            Maximum number of principal components to test (the method will
            evaluate values from 1 to ``pc_max-1``).
        noise : {'pipe','std','rms','astropy'}, optional
            Noise estimator to use (default 'pipe' which uses
            ``utils.pipe_mad``).
        plot : bool, optional
            If ``True``, display a plot of noise metric vs number of PCs.

        Returns
        -------
        pcs_to_include : ndarray
            Array of numbers of PCs tested (1..pc_max-1).
        mad_with_pcs : ndarray
            Noise metric computed for each tested number of PCs.
        fig : matplotlib.figure.Figure or None
            Figure object if ``plot=True``, else ``None``.
        axs : matplotlib.axes.Axes or None
            Axis object for the plot if produced, else ``None``.
        """
        # Choosing the method to compute noise
        if noise == 'pipe':
            noise_func = utils.pipe_mad
        elif noise == 'std':
            noise_func = np.nanstd
        elif noise == 'rms':
            noise_func = utils.rms
        elif noise == 'astropy':
            noise_func = lambda x: mad_std(x, ignore_nan=True)
        else:
            raise ValueError("Method should be one of 'pipe', 'std', 'rms', or 'astropy'.")
        
        pcs_to_include = np.arange(1, pc_max)
        mad_with_pcs = np.zeros(len(pcs_to_include))

        for p in tqdm(range(len(pcs_to_include))):
            # First compute the prediction
            x_pc = np.vstack(( np.ones(len(self.Psum)), self.PCA[0:int(pcs_to_include[p]),:] ))
            res_pc = np.linalg.lstsq(x_pc.T, self.Psum, rcond=None)
            co_pc = res_pc[0]
            pred_pc = np.dot(co_pc, x_pc)

            # Computing the MAD
            mad_with_pcs[p] = noise_func( self.Psum / pred_pc ) * 1e6

        # Number of PCs with minimum noise
        n_pc_of_min_scat = pcs_to_include[np.argmin(mad_with_pcs)]

        if plot:
            fig, axs = plt.subplots(figsize=(16/1.5, 9/1.5))
            axs.plot(pcs_to_include, mad_with_pcs, c='black')
            axs.axvline(n_pc_of_min_scat, color='r')
            
            axs.set_xlabel('Number of PCs included')
            axs.set_ylabel('Median Absolute Deviation')

            axs.set_xlim([ np.min(pcs_to_include), np.max(pcs_to_include) ])
            plt.grid()
        else:
            fig, axs = None, None

        return pcs_to_include, mad_with_pcs, fig, axs
    

class TESSData(object):
    """Helper to query and fetch TESS products from MAST and juliet.

    This lightweight utility class provides two convenience methods:
    ``get_lightcurves`` which delegates to ``juliet.utils.get_all_TESS_data``
    (when available) to gather per-instrument light curves, and
    ``get_tpfs`` which queries MAST for Target Pixel Files (TPFs),
    downloads the products and populates simple dictionaries on the
    instance with the extracted time and pixel-cube data.

    Parameters
    ----------
    object_name : str
        Target identifier accepted by MAST (for example a TIC or common
        object name).

    Attributes
    ----------
    object_name : str
        The provided target name.
    tim_lc, fl_lc, fle_lc : dict or None
        Time/flux/flux-error collections populated by
        ``get_lightcurves`` when that method is called.
    out_dict_lc : object or None
        Optional output dictionary returned by the juliet helper.
    tim_tpf, fl_tpf, fle_tpf, badpix, quality : dict
        Dictionaries populated by :meth:`get_tpfs` containing per-sector
        time arrays, pixel flux cubes, pixel flux error cubes, bad-pixel
        masks and quality arrays respectively.
    """

    def __init__(self, object_name):
        self.object_name = object_name

    def get_lightcurves(self, pdc=True, save=False, pout=None, **kwargs):
        """Collect TESS light curves for ``self.object_name``.

        This method attempts to use ``juliet.utils.get_all_TESS_data`` to
        retrieve light curves. The juliet helper may return either a
        simple tuple ``(tim, fl, fle)`` or an (out_dict, tim, fl, fle)
        tuple; both cases are handled and stored on the instance.

        Parameters
        ----------
        pdc : bool, optional
            If True request PDC-corrected fluxes when available. Default
            is ``True``.
        save : bool, optional
            If True the method will write ASCII files named ``LC_<object>_<inst>.dat``
            to ``pout``. Default ``False``.
        pout : str or None, optional
            Output directory used when ``save=True`` (defaults to the
            current working directory if ``None``).
        **kwargs : dict
            Forwarded to ``juliet.utils.get_all_TESS_data``.

        Returns
        -------
        tim_lc, fl_lc, fle_lc, out_dict_lc : tuple
            Stored time/flux/fluxerr containers and an optional output
            dictionary (``out_dict_lc`` may be ``None``).
        """
        try:
            self.tim_lc, self.fl_lc, self.fle_lc = juliet.utils.get_all_TESS_data(object_name=self.object_name, get_PDC=pdc, **kwargs)
            self.out_dict_lc = None
        except:
            self.out_dict_lc, self.tim_lc, self.fl_lc, self.fle_lc = juliet.utils.get_all_TESS_data(object_name=self.object_name, get_PDC=pdc, **kwargs)
        if save:
            for ins in self.tim.keys():
                fname = open( pout + '/LC_' + self.object_name + '_' + ins + '.dat', 'w' )
                for t in range( len(self.tim[ins]) ):
                    fname.write( str( self.tim[ins][t] ) + '\t' + str( self.fl[ins][t] ) + '\t' + str( self.fle[ins][t] ) + '\n' )
                fname.close()
        return self.tim_lc, self.fl_lc, self.fle_lc, self.out_dict_lc

    def get_tpfs(self, pout=None, load=False, save=False, savefits=False):
        """Download or load TESS Target Pixel Files (TPFs) for the target.

        If ``load`` is False the method queries MAST for TESS time-series
        products for ``self.object_name``, filters suitable products
        (TPFs), downloads them, reads the FITS tables and populates the
        instance dictionaries ``tim_tpf``, ``fl_tpf``, ``fle_tpf``,
        ``quality`` and ``badpix``. If ``save`` is True a small pickled
        dictionary per sector is written to ``pout``. ``load=True`` simply
        reads these previously saved dictionaries.

        Parameters
        ----------
        pout : str or None
            Output directory used when ``save=True``. If ``None`` the
            current working directory is used.
        load : bool, optional
            If True, attempt to load previously saved TPF pickles instead
            of querying/downloading from MAST. (Loading logic is not
            implemented in this helper; calling with ``load=True`` will
            bypass the MAST query branch.)
        save : bool, optional
            If True save extracted per-sector dictionaries as
            ``TPF_<object>_<sector>.pkl`` in ``pout``. Default ``False``.
        savefits : bool, optional
            If True, save the downloaded fits files in pout folder.
            Default is False.

        Returns
        -------
        tim_tpf, fl_tpf, fle_tpf, badpix, quality : tuple
            The dictionaries populated on the instance containing the
            per-sector time arrays, pixel flux cubes, flux error cubes,
            quality arrays and a placeholder bad-pixel mask.
        """
        if not load:
            try:
                obt = Observations.query_object(self.object_name, radius=0.01)
            except:
                raise Exception('The name of the object does not seem to be correct.\nPlease try again...')
            
            b = np.array([])
            for j in range(len(obt['intentType'])):
                if obt['obs_collection'][j] == 'TESS' and obt['dataproduct_type'][j] == 'timeseries':
                    b = np.hstack( (b, j) )
            
            if len(b) == 0:
                raise Exception('No TESS timeseries data available for this target (strange, right?!!).\nTry another target...')
            
            sectors, pi_name, obsids, exptime, new_b = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
            for i in range(len(b)):
                data1 = obt['dataURL'][int(b[i])]
                if data1[-9:] == 's_lc.fits':
                    fls = data1.split('-')
                    for j in range(len(fls)):
                        if len(fls[j]) == 5:
                            sec = fls[j]
                            tic = fls[j+1]
                    sectors = np.hstack((sectors, sec))
                    new_b = np.hstack((new_b, b[i]))
                    obsids = np.hstack((obsids, obt['obsid'][int(b[i])]))
                    pi_name = np.hstack((pi_name, obt['proposal_pi'][int(b[i])]))
                    exptime = np.hstack((exptime, obt['t_exptime'][int(b[i])]))
            print('Data products found over ' + str( len(sectors) ) + ' sectors.')
            print('Downloading them...')
            disp_tic, disp_sec = [], []
            
            # Dictionaries for saving the data in self
            self.tim_tpf, self.fl_tpf, self.fle_tpf, self.badpix, self.quality = {}, {}, {}, {}, {}

            for i in range(len(sectors)):
                dpr = Observations.get_product_list(obt[int(new_b[i])])
                cij = 0
                for j in range(len(dpr['obsID'])):
                    if dpr['description'][j] == 'Target pixel files':
                        cij = j
                
                tab = Observations.download_products(dpr[cij])
                lpt = tab['Local Path'][0][1:]
                
                # Reading fits
                hdul = fits.open(os.getcwd() + lpt)
                hdr = hdul[0].header
                
                ## Finding TIC-id and sector
                ticid = int(hdr['TICID'])
                ticid = f"{ticid:010}"
                disp_tic.append(ticid)

                sec_tess = 'TESS' + str( hdr['SECTOR'] )
                disp_sec.append(sec_tess)
                
                dta = Table.read(hdul[1])

                # Creating a mask array to mask NaN
                idx = np.ones( dta['FLUX'].shape[0], dtype=bool )
                for t in range( dta['FLUX'].shape[0] ):
                    nan = np.where( np.isnan( dta['FLUX'][t,:,:] ) )
                    if len( nan[0] ) == len( dta['FLUX'][t,:,:].flatten() ):
                        idx[t] = False

                self.tim_tpf[sec_tess] = dta['TIME'][idx]
                self.fl_tpf[sec_tess], self.fle_tpf[sec_tess] = dta['FLUX'][idx,:,:], dta['FLUX_ERR'][idx,:,:]
                self.quality[sec_tess] = dta['QUALITY'][idx]
                self.badpix[sec_tess] = np.ones( dta['FLUX'][idx,:,:].shape, dtype=bool )

                if save:
                    data_sector = {}
                    data_sector['TIME'] = dta['TIME'][idx]
                    data_sector['FLUX'], data_sector['FLUX_ERR'] = dta['FLUX'][idx,:,:], dta['FLUX_ERR'][idx,:,:]
                    data_sector['QUALITY'] = dta['QUALITY'][idx]
                    data_sector['BADPIX'] = np.ones( dta['FLUX'][idx,:,:].shape, dtype=bool )

                    if pout is None:
                        pout = os.getcwd()

                    ## And saving them
                    pickle.dump( data_sector, open(pout + '/TPF_' + self.object_name + '_' + sec_tess + '.pkl','wb') )
            
                if savefits:
                    if not Path(pout + '/TPF_' + self.object_name + '_fits').exists():
                        os.mkdir(pout + '/TPF_' + self.object_name + '_fits')
                    os.system('mv ' + os.getcwd() + lpt + ' ' + pout + '/TPF_' + self.object_name + '_fits/')
            
            # Delete the folder in the end
            os.system('rm -rf mastDownload')

        else:
            fnames = glob(pout + '/TPF_' + self.object_name + '_TESS*.pkl')
            
            # Dictionaries for saving the data in self
            self.tim_tpf, self.fl_tpf, self.fle_tpf, self.badpix, self.quality = {}, {}, {}, {}, {}
            for i in range(len(fnames)):
                data_sector = pickle.load( open( fnames[i], 'rb' ) )

                ## Sector name
                sec_name = fnames[i].split('/')[-1].split('.')[0].split('_')[-1]

                ## Loading the data
                self.tim_tpf[sec_name] = data_sector['TIME']
                self.fl_tpf[sec_name], self.fle_tpf[sec_name] = data_sector['FLUX'], data_sector['FLUX_ERR']
                self.quality[sec_name] = data_sector['QUALITY']
                self.badpix[sec_name] = data_sector['BADPIX']

    def ApPhoto(self, sector, aprad=None, sky_rad1=None, sky_rad2=None, brightpix=False, nos_brightest=12, nos_faintest=None, minmax=None):
        return ApPhoto(times=self.tim_tpf[sector], frames=self.fl_tpf[sector], errors=self.fle_tpf[sector], badpix=self.badpix[sector],\
                       aprad=aprad, sky_rad1=sky_rad1, sky_rad2=sky_rad2, brightpix=brightpix, nos_brightest=nos_brightest, nos_faintest=nos_faintest, minmax=minmax)


class KeplerData(object):
    def __init__(self, object_name):
        self.object_name = object_name
    
    def get_lightcurves(self, pdc=True, long_cadence=True, verbose=True, save=False, pout=None):
        """
        Collect Kepler/K2 light curves for ``self.object_name``.

        This method uses ``astroquery`` to retrieve light curves.
        The function returns a simple tuple ``(tim, fl, fle)``.

        Parameters
        ----------
        pdc : bool, optional
            If True request PDC-corrected fluxes when available. Default
            is ``True``.
        long_cadence : bool, optional
            If True the long cadence data will be downloaded. Default if False.
        verbose : bool
            Boolean on whether to print updates. Default is True
        save : bool, optional
            If True the method will write ASCII files named ``LC_<object>_<inst>.dat``
            to ``pout``. Default ``False``.
        pout : str or None, optional
            Output directory used when ``save=True`` (defaults to the
            current working directory if ``None``).
        """
        if ('K2' in self.object_name) and (not long_cadence):
            raise Exception('No Short Cadence data available for K2 objects.')
        try:
            obt = Observations.query_object(self.object_name, radius=0.001)
        except:
            raise Exception('The name of the object does not seem to be correct.\nPlease try again...')
        
        # b contains indices of the timeseries observations from TESS
        b = np.array([])
        for j in range(len(obt['intentType'])):
            if (obt['obs_collection'][j] == 'Kepler' or obt['obs_collection'][j] == 'K2') and obt['dataproduct_type'][j] == 'timeseries':
                b = np.hstack((b,j))
        
        if len(b) == 0:
            raise Exception('No Kepler/K2 timeseries data available for this target.\nTry another target...')
        
        # To extract obs-id from the observation table
        pi_name, obsids, exptime, scad, lcad = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        for i in range(len(b)):
            data1 = obt['dataURL'][int(b[i])]
            if 'lc' in data1.split('_'):
                lcad = np.hstack((lcad, b[i]))
            if 'sc' in data1.split('_'):
                scad = np.hstack((scad, b[i]))
            if 'llc.fits' in data1.split('_'):
                lcad = np.hstack((lcad, b[i]))
            pi_nm = obt['proposal_pi'][int(b[i])]
            if type(pi_nm) != str:
                pi_nm = 'K2 Team'
            pi_name = np.hstack((pi_name, pi_nm))
        
        if long_cadence:
            dwn_b = lcad
            keywd = 'Lightcurve Long Cadence'
        else:
            dwn_b = scad
            keywd = 'Lightcurve Short Cadence'
        
        for i in range(len(dwn_b)):
            obsids = np.hstack((obsids, obt['obsid'][int(dwn_b[i])]))
            exptime = np.hstack((exptime, obt['t_exptime'][int(dwn_b[i])]))
        
        disp_kic, disp_sec = [], []
        
        # Directory to save the data
        self.tim_lc, self.fl_lc, self.fle_lc = {}, {}, {}

        for i in range(len(dwn_b)):
            dpr = Observations.get_product_list(obt[int(dwn_b[i])])
            cij = []
            for j in range(len(dpr['obsID'])):
                if keywd in dpr['description'][j]:
                    cij.append(j)
            if verbose:
                print('Data products found over ' + str(len(cij)) + ' quarters/cycles.')
                print('Downloading them...')
            for j in range(len(cij)):
                sector = f"{i:02}" + f"{j:02}" + '-' + dpr['description'][cij[j]].split('- ')[1]
                
                # Downloading the data
                tab = Observations.download_products(dpr[cij[j]])
                lpt = tab['Local Path'][0][1:]
                # Reading fits
                hdul = fits.open(os.getcwd() + lpt)
                hdr = hdul[0].header
                kicid = int(hdr['KEPLERID'])
                kicid = f"{kicid:010}"
                dta = Table.read(hdul[1])
                # Available data products
                try:
                    if pdc:
                        fl = np.asarray(dta['PDCSAP_FLUX'])
                        fle = np.asarray(dta['PDCSAP_FLUX_ERR'])
                    else:
                        fl = np.asarray(dta['SAP_FLUX'])
                        fle = np.asarray(dta['SAP_FLUX_ERR'])
                except:
                    continue

                mask = np.isfinite(fl)                                # Creating Mask to remove Nans
                bjd1 = np.asarray(dta['TIME'])[mask] + hdul[1].header['BJDREFI']
                fl, fle = fl[mask], fle[mask]                         # Flux and Error in flux without Nans

                self.tim_lc[sector], self.fl_lc[sector], self.fle_lc[sector] = bjd1, fl / np.nanmedian(fl), fle / np.nanmedian(fl)
                
                disp_kic.append(kicid)
                disp_sec.append(sector)

        if verbose:
            print('----------------------------------------------------------------------------------------')
            print('Name\t\tKIC-id\t\tSector')
            print('----------------------------------------------------------------------------------------')
            for i in range(len(disp_kic)):
                print(self.object_name + '\t\t' + disp_kic[i] + '\t\t' + disp_sec[i])

        if save:
            for ins in self.tim_lc.keys():
                fname = open( pout + '/LC_' + self.object_name + '_' + ins + '.dat', 'w' )
                for t in range( len(self.tim_lc[ins]) ):
                    fname.write( str( self.tim_lc[ins][t] ) + '\t' + str( self.fl_lc[ins][t] ) + '\t' + str( self.fle_lc[ins][t] ) + '\n' )
                fname.close()

        # Deleting the data
        os.system('rm -rf mastDownload')

        return self.tim_lc, self.fl_lc, self.fle_lc
    
    def get_tpfs(self, long_cadence=True, load=False, verbose=True, save=False, pout=None, savefits=False):
        """
        Collect Kepler/K2 target pixel files for ``self.object_name``.

        This method uses ``astroquery`` to retrieve target pixel files.

        Parameters
        ----------
        load : bool, optional
            If True request PDC-corrected fluxes when available. Default
            is ``True``.
        long_cadence : bool, optional
            If True the long cadence data will be downloaded. Default if False.
        verbose : bool
            Boolean on whether to print updates. Default is True
        save : bool, optional
            If True the method will write ASCII files named ``LC_<object>_<inst>.dat``
            to ``pout``. Default ``False``.
        pout : str or None, optional
            Output directory used when ``save=True`` (defaults to the
            current working directory if ``None``).
        """
        if not load:
            if ('K2' in self.object_name) and (not long_cadence):
                raise Exception('No Short Cadence data available for K2 objects.')
            try:
                obt = Observations.query_object(self.object_name, radius=0.001)
            except:
                raise Exception('The name of the object does not seem to be correct.\nPlease try again...')
            
            # b contains indices of the timeseries observations from TESS
            b = np.array([])
            for j in range(len(obt['intentType'])):
                if (obt['obs_collection'][j] == 'Kepler' or obt['obs_collection'][j] == 'K2') and obt['dataproduct_type'][j] == 'timeseries':
                    b = np.hstack((b,j))
            
            if len(b) == 0:
                raise Exception('No Kepler/K2 timeseries data available for this target.\nTry another target...')
            
            # To extract obs-id from the observation table
            pi_name, obsids, exptime, scad, lcad = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
            for i in range(len(b)):
                data1 = obt['dataURL'][int(b[i])]
                if 'lc' in data1.split('_'):
                    lcad = np.hstack((lcad, b[i]))
                if 'sc' in data1.split('_'):
                    scad = np.hstack((scad, b[i]))
                if 'llc.fits' in data1.split('_'):
                    lcad = np.hstack((lcad, b[i]))
                if 'slc.fits' in data1.split('_'):
                    scad = np.hstack((scad, b[i]))
                pi_nm = obt['proposal_pi'][int(b[i])]
                if type(pi_nm) != str:
                    pi_nm = 'K2 Team'
                pi_name = np.hstack((pi_name, pi_nm))
            
            if long_cadence:
                dwn_b = lcad
                keywd = 'Target Pixel Long Cadence'
            else:
                dwn_b = scad
                keywd = 'Target Pixel Short Cadence'
            
            for i in range(len(dwn_b)):
                obsids = np.hstack((obsids, obt['obsid'][int(dwn_b[i])]))
                exptime = np.hstack((exptime, obt['t_exptime'][int(dwn_b[i])]))
            
            disp_kic, disp_sec = [], []
            
            # Dictionaries to save the data
            self.tim_tpf, self.fl_tpf, self.fle_tpf, self.badpix, self.quality = {}, {}, {}, {}, {}

            for i in range(len(dwn_b)):
                dpr = Observations.get_product_list(obt[int(dwn_b[i])])
                cij = []
                for j in range(len(dpr['obsID'])):
                    if keywd in dpr['description'][j]:
                        cij.append(j)
                if verbose:
                    print('Data products found over ' + str(len(cij)) + ' quarters/cycles.')
                    print('Downloading them...')

                for j in range(len(cij)):
                    sector = f"{i:02}" + f"{j:02}" + '-' + dpr['description'][cij[j]].split('- ')[1]
                    
                    # Downloading the data
                    tab = Observations.download_products(dpr[cij[j]])
                    lpt = tab['Local Path'][0][1:]

                    # Unzipping the file
                    os.system('gunzip --keep ' + os.getcwd() + lpt)

                    # Reading fits
                    hdul = fits.open(os.getcwd() + lpt[:-3])
                    hdr = hdul[0].header
                    kicid = int(hdr['KEPLERID'])
                    kicid = f"{kicid:010}"
                    dta = Table.read(hdul[1])

                    # Creating a mask array to mask NaN
                    idx = np.ones( dta['FLUX'].shape[0], dtype=bool )
                    for t in range( dta['FLUX'].shape[0] ):
                        nan = np.where( np.isnan( dta['FLUX'][t,:,:] ) )
                        if len( nan[0] ) == len( dta['FLUX'][t,:,:].flatten() ):
                            idx[t] = False

                    self.tim_tpf[sector] = dta['TIME'][idx] + hdul[1].header['BJDREFI']
                    self.fl_tpf[sector], self.fle_tpf[sector] = dta['FLUX'][idx,:,:], dta['FLUX_ERR'][idx,:,:]
                    self.quality[sector] = dta['QUALITY'][idx]
                    self.badpix[sector] = np.ones( dta['FLUX'][idx,:,:].shape, dtype=bool )

                    if save:
                        data_sector = {}
                        data_sector['TIME'] = dta['TIME'][idx] + hdul[1].header['BJDREFI']
                        data_sector['FLUX'], data_sector['FLUX_ERR'] = dta['FLUX'][idx,:,:], dta['FLUX_ERR'][idx,:,:]
                        data_sector['QUALITY'] = dta['QUALITY'][idx]
                        data_sector['BADPIX'] = np.ones( dta['FLUX'][idx,:,:].shape, dtype=bool )

                        if pout is None:
                            pout = os.getcwd()

                        ## And saving them
                        pickle.dump( data_sector, open(pout + '/TPF_' + self.object_name + '_Kep' + sector + '.pkl','wb') )
                    
                    disp_kic.append(kicid)
                    disp_sec.append(sector)

                    if savefits:
                        if not Path(pout + '/TPF_' + self.object_name + '_fits').exists():
                            os.mkdir(pout + '/TPF_' + self.object_name + '_fits')
                        os.system('mv ' + os.getcwd() + lpt + ' ' + pout + '/TPF_' + self.object_name + '_fits')
                        os.system('mv ' + os.getcwd() + lpt[:-3] + ' ' + pout + '/TPF_' + self.object_name + '_fits')
            
            # Delete the folder in the end
            os.system('rm -rf mastDownload')

            if verbose:
                print('----------------------------------------------------------------------------------------')
                print('Name\t\tKIC-id\t\tSector')
                print('----------------------------------------------------------------------------------------')
                for i in range(len(disp_kic)):
                    print(self.object_name + '\t\t' + disp_kic[i] + '\t\t' + disp_sec[i])
        else:
            fnames = glob(pout + '/TPF_' + self.object_name + '_Kep*.pkl')
            
            # Dictionaries for saving the data in self
            self.tim_tpf, self.fl_tpf, self.fle_tpf, self.badpix, self.quality = {}, {}, {}, {}, {}
            for i in range(len(fnames)):
                data_sector = pickle.load( open( fnames[i], 'rb' ) )

                ## Sector name
                sec_name = fnames[i].split('/')[-1].split('.')[0].split('_')[-1]

                ## Loading the data
                self.tim_tpf[sec_name] = data_sector['TIME']
                self.fl_tpf[sec_name], self.fle_tpf[sec_name] = data_sector['FLUX'], data_sector['FLUX_ERR']
                self.quality[sec_name] = data_sector['QUALITY']
                self.badpix[sec_name] = data_sector['BADPIX']

    def ApPhoto(self, sector, aprad=None, sky_rad1=None, sky_rad2=None, brightpix=False, nos_brightest=12, nos_faintest=None, minmax=None):
        return ApPhoto(times=self.tim_tpf[sector], frames=self.fl_tpf[sector], errors=self.fle_tpf[sector], badpix=self.badpix[sector],\
                       aprad=aprad, sky_rad1=sky_rad1, sky_rad2=sky_rad2, brightpix=brightpix, nos_brightest=nos_brightest, nos_faintest=nos_faintest, minmax=minmax)