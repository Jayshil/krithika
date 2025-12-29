import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gd
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d
import plotstyles
import juliet
import utils

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
    """
    Given the `juliet` input folder, this class can create various plots
    """
    def __init__(self, input_folder, N=1000, **kwargs):
        # First order of business: loading the juliet folder
        self.dataset = juliet.load(input_folder=input_folder)
        self.res = self.dataset.fit(**kwargs)

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

    
    def phase_folded_lc(self, phmin=0.8, instruments=None, highres=False, nrandom=50, quantile_models=True, one_plot=None, figsize=(16/1.5, 9/1.5), pycheops_binning=False):
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
                    ax1.fill_between(phs_model, y1=lo_68CI*ppt, y2=up_68CI*ppt, color='orangered', alpha=0.5, zorder=25)
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
                    nbin = int( np.sum(idx_transits) / 20 )

                    bin_phs_tra, bin_fl_tra, bin_fle_tra = juliet.utils.bin_data(x=phs[idx_transits], y=detrend_data[idx_transits]*ppt, n_bin=nbin, yerr=detrend_errs[idx_transits]*ppt)
                    if phasecurve or eclipse:
                        _, bin_res_tra, bin_reserr_tra = juliet.utils.bin_data(x=phs[idx_transits], y=residuals[idx_transits]*ppt*1e6, n_bin=nbin, yerr=detrend_errs[idx_transits]*ppt)
                    else:
                        # When transit only, we haven't multiplied the errorbars with 1e6 yet; so we need to do it again
                        _, bin_res_tra, bin_reserr_tra = juliet.utils.bin_data(x=phs[idx_transits], y=residuals[idx_transits]*1e6, n_bin=nbin, yerr=detrend_errs[idx_transits]*1e6)
                else:
                    binwid = np.ptp( phs[idx_transits] ) / 20.

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
                        ax3.fill_between(phs_model, y1=lo_68CI, y2=up_68CI, color='orangered', alpha=0.5, zorder=25)
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
                        bin_phs_pc, bin_fl_pc, bin_fle_pc = juliet.utils.bin_data(x=phs, y=detrend_data, n_bin=int( len(phs)/40 ), yerr=detrend_errs)
                        _, bin_res_pc, bin_reserr_pc = juliet.utils.bin_data(x=phs, y=residuals*1e6, n_bin=int( len(phs)/40 ), yerr=detrend_errs)
                    else:
                        bin_phs_pc, bin_fl_pc, bin_fle_pc, _ = utils.lcbin(time=phs, flux=detrend_data, binwidth=1/50)
                        _, bin_res_pc, bin_reserr_pc, _ = utils.lcbin(time=phs, flux=residuals*1e6, binwidth=1/50)

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
                nbin = int( np.sum(idx_transits) / 20 )

                # Performing binning
                bin_phs_tra, bin_fl_tra, bin_fle_tra = juliet.utils.bin_data(x=bin_phs[idx_transits], y=bin_fl[idx_transits]*ppt, n_bin=nbin, yerr=bin_fle[idx_transits]*ppt)
                if phasecurve or eclipse:
                    _, bin_res_tra, bin_reserr_tra = juliet.utils.bin_data(x=bin_phs[idx_transits], y=bin_res[idx_transits]*ppt*1e6, n_bin=nbin, yerr=bin_fle[idx_transits]*ppt)
                else:
                    # When transit only, we haven't multiplied the errorbars with 1e6 yet; so we need to do it again
                    _, bin_res_tra, bin_reserr_tra = juliet.utils.bin_data(x=bin_phs[idx_transits], y=bin_res[idx_transits]*1e6, n_bin=nbin, yerr=bin_fle[idx_transits]*1e6)
            else:
                binwid = np.ptp( bin_phs[idx_transits] ) / 20.

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
                    bin_phs_pc, bin_fl_pc, bin_fle_pc = juliet.utils.bin_data(x=bin_phs, y=bin_fl, n_bin=int( len(phs)/20 ), yerr=bin_fle)
                    _, bin_res_pc, bin_reserr_pc = juliet.utils.bin_data(x=bin_phs, y=bin_res*1e6, n_bin=int( len(phs)/20 ), yerr=bin_fle)
                else:
                    bin_phs_pc, bin_fl_pc, bin_fle_pc, _ = utils.lcbin(time=bin_phs, flux=bin_fl, binwidth=1/50)
                    _, bin_res_pc, bin_reserr_pc, _ = utils.lcbin(time=bin_phs, flux=bin_res*1e6, binwidth=1/50)

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