import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gd
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d
import plotstyles
import juliet

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
    
