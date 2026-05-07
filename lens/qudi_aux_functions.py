# This is for functions that you use repeatedly during the analysis of the data
# that are compatible with the qudi format for storing data and for fitting data 

import numpy as np
import matplotlib.pyplot as plt

from lens.analysisFunctions import *

##
## Opening data
##
def qudi_open_dat(filename,show_plots=False,show_header=False,show_bare_laser=True):
    # Read header
    with open(filename) as f:
        lines = f.readlines()
        end_of_header = [i for i,line in enumerate(lines) if ('---- END HEADER ----' in line)][0]
        header = ''.join(lines[:end_of_header])
        if show_header:
            print(header)
    # Read data
    data = np.loadtxt(fname=filename,unpack=True)
    # Plot data
    if show_plots:
        plt.plot(data[0],data[1])
        plt.ylabel('Volts')
        plt.xlabel('mw frequency [Hz]')
        plt.show()
        if show_bare_laser:
            plt.plot(data[0],data[2])
            plt.ylabel('Volts')
            plt.xlabel('mw frequency [Hz]')
            plt.show()
    return header,data

def qudi_open_raw_data(filename,show_header=False):
    # Read header
    with open(filename) as f:
        lines = f.readlines()
        end_of_header = [i for i,line in enumerate(lines) if ('---- END HEADER ----' in line)][0]
        header = ''.join(lines[:end_of_header])
        if show_header:
            print(header)
    # Read data
    data = np.loadtxt(fname=filename,unpack=True)
    return header,data

# Normalize data:
def qudi_normalize_raw_data(fname_Pl,fname_laser,show_plots=False):
    """
    Normalize data_Pl/data_laser
    """
    # Read data
    data_pl = np.loadtxt(fname=fname_Pl,unpack=True)
    data_laser = np.loadtxt(fname=fname_laser,unpack=True)
    new_data = data_pl.copy()
    new_data[1:] = -data_pl[1:]/data_laser[1:]
    if show_plots:
        for i in range(1,new_data.shape[0]):
            plt.plot(new_data[0],new_data[i])
        plt.ylabel('arb. units')
        plt.xlabel('mw frequency [Hz]')
        plt.title('normalized data')
        plt.show()
    return new_data 



hyperfine = 2.16e6
amplitude_scaling = np.array([1/3,2/3,1,2/3,1/3]) #np.array([1,2,3,2,1]) #
#from lens.analysisFunctions import funcLorentz
# Instead of using the lens.analysisFunctions.funcLorentz version, we will use the funcLorentz where a is the amplitude (independent parameters).
# This is better because the other option uses area, which depends on the width.

########################################################################
## This is a copy of the functions, estimators and models defined in  
## /home/nvuser/qudi/modules/util/fit_models/odmr_spectra.py
########################################################################
from lens.qudi_link.util.fit_models.model import FitModelBase, estimator
from lens.qudi_link.util.fit_models.helpers import correct_offset_histogram, smooth_data, sort_check_data
#from qudi.util.fit_models.helpers import estimate_double_peaks, estimate_triple_peaks
#from qudi.util.fit_models.linear import Linear

amplitude_scaling = np.array([1/3,2/3,1,2/3,1/3]) #np.array([1,2,3,2,1]) #

def funcLorentz(x,y0,a,xc,w):
    """
    y0 + a*w**2/(4*(x-xc)**2+w**2)
    Note:    a-->amplitude  w-->FWHM""" #"""returns y0 + a*(2/np.pi)*w/(4*(x-xc)**2+w**2)"""
    return y0 + a*w**2/(4*(x-xc)**2+w**2) #y0 + a*(2/np.pi)*w/(4*(x-xc)**2+w**2)
def funcDerivativeLorentz_naive(x,y0,a,xc,w):
    """returns y0 - a*w**2*8*(x-xc)/(4*(x-xc)**2+w**2)**2
    Note:    a-->amplitude(of original lorentzian)  w-->FWHM""" # """returns y0 - a*(2/np.pi)*w*8*(x-xc)/(4*(x-xc)**2+w**2)**2"""
    return y0 - a*w**2*8*(x-xc)/(4*(x-xc)**2+w**2)**2 # y0 - a*(2/np.pi)*w*8*(x-xc)/(4*(x-xc)**2+w**2)**2
def funcDerivativeLorentz(x,y0,a,xc,w):
    """returns y0 - 16*a*w**3*(x-xc)/((x-xc)**2+3*w**2)**2
    Note: where a is the maximum amplitude (or minimum) of the dispersion signal
    that happens at funcDerivativeLorentz[x-xc = +-w]
    The conversion back to the funcDerivativeLorentz_naive parameters is given by:
    w_disp = fwhm_lorentz/np.sqrt(12)
    a_disp = (3*a_lor)/(8*w_disp)"""
    return y0 - 16*a*w**3*(x-xc)/((x-xc)**2+3*w**2)**2

    
    
def func3Lorentz(x,y0,a,xc,w,a1,xc1,w1,a2,xc2,w2):
    return y0 + funcLorentz(x,0,a,xc,w) + funcLorentz(x,0,a1,xc1,w1) + funcLorentz(x,0,a2,xc2,w2)
def func6Lorentz(x,y0,a,xc,w, a1,xc1,w1, a2,xc2,w2, a3,xc3,w3, a4,xc4,w4, a5,xc5,w5):
    return y0 + func3Lorentz(x,0,a,xc,w,a1,xc1,w1,a2,xc2,w2) + func3Lorentz(x,0,a3,xc3,w3, a4,xc4,w4, a5,xc5,w5)
def func5Lorentz(x,y0,a,xc,w, a1,xc1,w1, a2,xc2,w2, a3,xc3,w3, a4,xc4,w4):
    return y0 + funcLorentz(x,0,a,xc,w) + funcLorentz(x,0,a1,xc1,w1) + funcLorentz(x,0,a2,xc2,w2) + funcLorentz(x,0,a3,xc3,w3) + funcLorentz(x,0,a4,xc4,w4)
def func10Lorentz(x,y0,a,xc,w, a1,xc1,w1, a2,xc2,w2, a3,xc3,w3, a4,xc4,w4, a5,xc5,w5,   a6,xc6,w6, a7,xc7,w7, a8,xc8,w8, a9,xc9,w9):
    return func5Lorentz(x,y0,a,xc,w, a1,xc1,w1, a2,xc2,w2, a3,xc3,w3, a4,xc4,w4) + func5Lorentz(x,0,a5,xc5,w5, a6,xc6,w6, a7,xc7,w7, a8,xc8,w8, a9,xc9,w9)
def func7Lorentz(x,y0,a,xc,w, a1,xc1,w1, a2,xc2,w2, a3,xc3,w3, a4,xc4,w4,a5,xc5,w5, a6,xc6,w6):
    return y0 + funcLorentz(x,0,a,xc,w) + funcLorentz(x,0,a1,xc1,w1) + funcLorentz(x,0,a2,xc2,w2) + funcLorentz(x,0,a3,xc3,w3) + funcLorentz(x,0,a4,xc4,w4) + funcLorentz(x,0,a5,xc5,w5) + funcLorentz(x,0,a6,xc6,w6)

def func3DerivativeLorentz(x,y0,a,xc,w,a1,xc1,w1,a2,xc2,w2):
    return y0 + funcDerivativeLorentz(x,0,a,xc,w) + funcDerivativeLorentz(x,0,a1,xc1,w1) + funcDerivativeLorentz(x,0,a2,xc2,w2)
def func6DerivativeLorentz(x,y0,a,xc,w, a1,xc1,w1, a2,xc2,w2, a3,xc3,w3, a4,xc4,w4, a5,xc5,w5):
    return y0 + func3DerivativeLorentz(x,0,a,xc,w,a1,xc1,w1,a2,xc2,w2) + func3DerivativeLorentz(x,0,a3,xc3,w3, a4,xc4,w4, a5,xc5,w5)
def func5DerivativeLorentz(x,y0,a,xc,w, a1,xc1,w1, a2,xc2,w2, a3,xc3,w3, a4,xc4,w4):
    return y0 + funcDerivativeLorentz(x,0,a,xc,w) + funcDerivativeLorentz(x,0,a1,xc1,w1) + funcDerivativeLorentz(x,0,a2,xc2,w2) + funcDerivativeLorentz(x,0,a3,xc3,w3) + funcDerivativeLorentz(x,0,a4,xc4,w4)
def func10DerivativeLorentz(x,y0,a,xc,w, a1,xc1,w1, a2,xc2,w2, a3,xc3,w3, a4,xc4,w4,a5,xc5,w5,   a6,xc6,w6, a7,xc7,w7, a8,xc8,w8, a9,xc9,w9):
    return func5DerivativeLorentz(x,y0,a,xc,w, a1,xc1,w1, a2,xc2,w2, a3,xc3,w3, a4,xc4,w4) + func5DerivativeLorentz(x,0,a5,xc5,w5, a6,xc6,w6, a7,xc7,w7, a8,xc8,w8, a9,xc9,w9)
def func7DerivativeLorentz(x,y0,a,xc,w, a1,xc1,w1, a2,xc2,w2, a3,xc3,w3, a4,xc4,w4, a5,xc5,w5,a6,xc6,w6):
    return func6DerivativeLorentz(x,y0,a,xc,w, a1,xc1,w1, a2,xc2,w2, a3,xc3,w3, a4,xc4,w4, a5,xc5,w5) + funcDerivativeLorentz(x,0,a6,xc6,w6)

########################################################################
########################################################################

def single_peak_estimator(data, x, useMinMaxCenter=True):
        data, x = sort_check_data(data, x)
        # Smooth data
        filter_width = max(1, int(round(len(x) / 30))) # int(round(len(x) / 20)))
        data_smoothed, _ = smooth_data(data, filter_width)
        data_smoothed, offset = correct_offset_histogram(data_smoothed, bin_width=2 * filter_width)

        if useMinMaxCenter:
            # determine peak position
            center = x[np.argmax(data_smoothed)]
        else:
            # determine central zero crossing by maximizing correlation between y and y reversed.
            # x, y must be 1D arrays (x strictly increasing and uniform (constant dx))
            # 
            n = data_smoothed.size
            # zero-mean optionally helps (not strictly necessary)
            yz = data_smoothed - offset
            # target is the reversed signal
            y_rev = yz[::-1]
            # next power of two for FFT speed (optional)
            m = int(2**np.ceil(np.log2(2*n - 1)))
            # FFT-based cross-correlation
            fy = np.fft.rfft(yz, n=m)
            fry = np.fft.rfft(y_rev, n=m)
            cc = np.fft.irfft(fy * np.conj(fry), n=m)
            # valid lags run from -(n-1) .. (n-1)
            cc = np.concatenate([cc[-(n-1):], cc[:n]])  # length 2n-1
            lags = np.arange(-(n-1), n)
            # find lag of max correlation
            idx = np.argmax(cc)
            lag = lags[idx]   # lag in index units: positive means y leads y_rev
            # Convert lag to shift in x: shift = lag * dx 
            dx = x[1] - x[0]
            shift = lag * dx   # sign: adjust so shift maps to center location
            # For reversed comparison the center is at midpoint:
            center = (x[0] + x[-1]) / 2 + shift / 2

        # calculate amplitude
        #amplitude = abs(max(data_smoothed))
        max_index = np.argmax(abs(data_smoothed)) # data_smoothed is without offset
        amplitude = data_smoothed[max_index]
        
        # according to the derived formula, calculate FWHM. The crucial part is here that the
        # offset was estimated correctly, then the area under the curve is calculated correctly:
        numerical_integral = np.trapz(data_smoothed, x)
        fwhm = 2*abs(numerical_integral / (np.pi * amplitude))
        # the 2 factor is because in funcLorentz(x,y0,a,xc,w), the w is the FWHM (instead of the HWHM of the qudi lorentzian model)
        
        # get useful bounds
        x_spacing = min(abs(np.ediff1d(x)))
        x_span = abs(x[-1] - x[0])
        data_span = abs(max(data) - min(data))
        
        return offset,amplitude,center,fwhm,[x_spacing,x_span,data_span]

def single_dispersion_estimator(data, x, useMinMaxCenter=True):
        data, x = sort_check_data(data, x)
        
        # Smooth data
        filter_width = max(1, int(round(len(x) / 30))) # int(round(len(x) / 20)))
        data_smoothed, _ = smooth_data(data, filter_width)
        
        # determine offset (assuming symmetric dispersion signal)
        offset = np.mean(data_smoothed)

        if useMinMaxCenter:
            # determine zero crossing (assuming symmetric dispersion signal)
            center = (x[np.argmax(data_smoothed)] + x[np.argmin(data_smoothed)])/2
        else:
            # determine central zero crossing by maximizing correlation between y and -y reversed.
            # x, y must be 1D arrays (x strictly increasing and uniform (constant dx))
            # 
            n = data_smoothed.size
            # zero-mean optionally helps (not strictly necessary)
            yz = data_smoothed - offset
            # target is negative reversed signal
            y_rev_neg = -yz[::-1]
            # next power of two for FFT speed (optional)
            m = int(2**np.ceil(np.log2(2*n - 1)))
            # FFT-based cross-correlation
            fy = np.fft.rfft(yz, n=m)
            fry = np.fft.rfft(y_rev_neg, n=m)
            cc = np.fft.irfft(fy * np.conj(fry), n=m)
            # valid lags run from -(n-1) .. (n-1)
            cc = np.concatenate([cc[-(n-1):], cc[:n]])  # length 2n-1
            lags = np.arange(-(n-1), n)
            # find lag of max correlation
            idx = np.argmax(cc)
            lag = lags[idx]   # lag in index units: positive means y leads y_rev_neg
            # Convert lag to shift in x: shift = lag * dx 
            dx = x[1] - x[0]
            shift = lag * dx   # sign: adjust so shift maps to center location
            # For reversed comparison the center is at midpoint:
            center = (x[0] + x[-1]) / 2 + shift / 2
        
        # calculate amplitude
        #amplitude = ( abs(max(data_smoothed)) + abs(min(data_smoothed)) )/2
        amplitude = ( abs(max(data)) + abs(min(data)) )/2
        # we don't use the smoothed data for the amplitude because the narrow lines
        # result in an under estimation

        
        # calculate width
        if useMinMaxCenter:
            width = abs(x[np.argmax(data_smoothed)] - x[np.argmin(data_smoothed)])/2
        else:
            center_index = len(data_smoothed)//2 + lag//2
            # correct the case of first/last point as center
            if center_index==0 or center_index==len(data_smoothed):
                center_index = len(data_smoothed)//2
            # find first min/max after the center
            data_smoothed[center_index:]
            w_shift_r = max((np.diff(data_smoothed[center_index:])>0).argmax(),
                            (np.diff(data_smoothed[center_index:])<0).argmax())
            # find first min/max before the center
            w_shift_l = max((np.diff(data_smoothed[:center_index])>0).argmax(),
                            (np.diff(data_smoothed[:center_index])<0).argmax())
            # 
            width = abs(np.mean([w_shift_r*dx,w_shift_l*dx]))
        
        # get useful bounds
        x_spacing = min(abs(np.ediff1d(x)))
        x_span = abs(x[-1] - x[0])
        data_span = abs(max(data) - min(data))
        
        return offset,amplitude,center,width,[x_spacing,x_span,data_span]
        
        ################
        # This section was useful when the funcDerivativeLorentz_naive, now we use funcDerivativeLorentz so we directly estimate amplitude and width
        ################
        ## calculate amplitude (assuming derivative of a peak signal starting from 0)
        ## We use the numerical integral based on the Trapezoidal rule https://en.wikipedia.org/wiki/Trapezoidal_rule
        #numerical_integral_1 = np.cumsum((data_smoothed[1:]+data_smoothed[:-1])/2)*x_spacing
        #amplitude = abs(max(numerical_integral_1))
        #
        ## according to the derived formula, calculate FWHM. The crucial part is here that the
        ## offset was estimated correctly, then the area under the curve is calculated correctly:
        ## NOTE that we integrate (trapz is numerical integral) twice since we are using the Lorentzian derivative
        #numerical_integral_2 = np.trapz(numerical_integral_1, x[1:])
        #fwhm = 2*abs(numerical_integral_2 / (np.pi * amplitude))
        ## the 2 factor is because in funcLorentz(x,y0,a,xc,w), the w is the FWHM (instead of the HWHM of the qudi lorentzian model)
        #
        #return offset,amplitude,center,fwhm,[x_spacing,x_span,data_span]
        ################
        
########################################################################
########################################################################

class ODMR1PeakSpectrum(FitModelBase):
    """ Single peak Lorentzian  """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_param_hint('offset', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('center', value=1e-15, min=0, max=np.inf)
        self.set_param_hint('fwhm', value=0., min=0, max=np.inf)

    @staticmethod
    def _model_function(x, offset,amplitude,center,fwhm):
        return funcLorentz(x,offset,amplitude,center,fwhm)
        
    @estimator('Peak')
    def estimate_peak(self, data, x):
        data, x = sort_check_data(data, x)
        offset,amplitude,center,fwhm,[x_spacing,x_span,data_span] = single_peak_estimator(data, x)
        
        estimate = self.make_params()
        estimate['amplitude'].set(value=amplitude, min=0, max=100 * amplitude)
        estimate['fwhm'].set(value=fwhm, min=x_spacing/10, max=x_span)
        estimate['center'].set(value=center, min=min(x) - x_span / 2, max=max(x) + x_span / 2)
        estimate['offset'].set(value=offset, min=min(data) - data_span / 2, max=max(data) + data_span / 2 )
        return estimate
        
    @estimator('Dip')
    def estimate_dip(self, data, x):
        estimate = self.estimate_peak(-data, x)
        estimate['offset'].set(value=-estimate['offset'].value,min=-estimate['offset'].max,max=-estimate['offset'].min)
        estimate['amplitude'].set(value=-estimate['amplitude'].value,min=-estimate['amplitude'].max,max=-estimate['amplitude'].min)
        return estimate

    @estimator('Peak (no offset)')
    def estimate_peak_no_offset(self, data, x):
        estimate = self.estimate_peak(data, x)
        estimate['offset'].set(value=0, min=-np.inf, max=np.inf, vary=False)
        return estimate

    @estimator('Dip (no offset)')
    def estimate_dip_no_offset(self, data, x):
        estimate = self.estimate_dip(data, x)
        estimate['offset'].set(value=0, min=-np.inf, max=np.inf, vary=False)
        return estimate

class ODMR3PeakSpectrum(FitModelBase):
    """ Triple peak Lorentzian """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_param_hint('offset', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('center', value=1e-15, min=0, max=np.inf)
        self.set_param_hint('fwhm', value=0., min=0, max=np.inf)
        self.set_param_hint('hyperfine', value=2.16e6, min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude_factor_1', value=1, min=0, max=np.inf)

    @staticmethod
    def _model_function(x, offset,amplitude,center,fwhm,hyperfine,amplitude_factor_1):
        xc0 = center - hyperfine
        xc1 = center
        xc2 = center + hyperfine
        return func3Lorentz(x,offset,amplitude*amplitude_factor_1,xc0,fwhm, amplitude,xc1,fwhm, amplitude*amplitude_factor_1,xc2,fwhm)
    
    @estimator('Peak (center self correl)')
    def estimate_peak_selfcorr(self, data, x):
        data, x = sort_check_data(data, x)
        offset,amplitude,center,fwhm,[x_spacing,x_span,data_span] = single_peak_estimator(data, x, useMinMaxCenter=False)
        
        # correct since it was obtained from the area assuming a single peak
        fwhm = fwhm/3
        
        estimate = self.make_params()
        estimate['amplitude'].set(value=amplitude, min=0, max=100 * amplitude)
        estimate['fwhm'].set(value=fwhm, min=x_spacing/10, max=x_span)
        estimate['center'].set(value=center, min=min(x) - x_span / 2, max=max(x) + x_span / 2)
        estimate['offset'].set(value=offset, min=min(data) - data_span / 2, max=max(data) + data_span / 2 )
        estimate['amplitude_factor_1'].set(value=1, min=0, max=np.inf)
        return estimate
        
    @estimator('Peak')
    def estimate_peak(self, data, x):
        data, x = sort_check_data(data, x)
        offset,amplitude,center,fwhm,[x_spacing,x_span,data_span] = single_peak_estimator(data, x)
        
        # correct since it was obtained from the area assuming a single peak
        fwhm = fwhm/3
        
        estimate = self.make_params()
        estimate['amplitude'].set(value=amplitude, min=0, max=100 * amplitude)
        estimate['fwhm'].set(value=fwhm, min=x_spacing/10, max=x_span)
        estimate['center'].set(value=center, min=min(x) - x_span / 2, max=max(x) + x_span / 2)
        estimate['offset'].set(value=offset, min=min(data) - data_span / 2, max=max(data) + data_span / 2 )
        estimate['amplitude_factor_1'].set(value=1, min=0, max=np.inf)
        return estimate
        
    @estimator('Peak (no offset)')
    def estimate_peak_no_offset(self, data, x):
        estimate = self.estimate_peak(data, x)
        estimate['offset'].set(value=0, min=-np.inf, max=np.inf, vary=False)
        return estimate

class ODMR6PeakSpectrum(FitModelBase):
    """ Six peak Lorentzian """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_param_hint('offset', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('center', value=1e-15, min=0, max=np.inf)
        self.set_param_hint('fwhm', value=0., min=0, max=np.inf)
        self.set_param_hint('hyperfine', value=2.16e6, min=-np.inf, max=np.inf)
        self.set_param_hint('mod_rate', value=4.5e6, min=-np.inf, max=np.inf)

    @staticmethod
    def _model_function(x, offset,amplitude,center,fwhm,hyperfine,mod_rate):
        xc0 = center - mod_rate - hyperfine
        xc1 = center - mod_rate
        xc2 = center - mod_rate + hyperfine
        xc3 = center + mod_rate - hyperfine
        xc4 = center + mod_rate
        xc5 = center + mod_rate + hyperfine
        return func6Lorentz(x,offset,amplitude,xc0,fwhm, amplitude,xc1,fwhm, amplitude,xc2,fwhm, amplitude,xc3,fwhm, amplitude,xc4,fwhm, amplitude,xc5,fwhm)
    
    @estimator('Peak')
    def estimate_peak(self, data, x):
        data, x = sort_check_data(data, x)
        offset,amplitude,center,fwhm,[x_spacing,x_span,data_span] = single_peak_estimator(data, x)
        
        # correct since it was obtained from the area assuming a single peak
        fwhm = fwhm/6
        
        estimate = self.make_params()
        estimate['amplitude'].set(value=amplitude, min=0, max=100 * amplitude)
        estimate['fwhm'].set(value=fwhm, min=x_spacing/10, max=x_span)
        estimate['center'].set(value=center, min=min(x) - x_span / 2, max=max(x) + x_span / 2)
        estimate['offset'].set(value=offset, min=min(data) - data_span / 2, max=max(data) + data_span / 2 )
        return estimate
        
    @estimator('Dip')
    def estimate_dip(self, data, x):
        estimate = self.estimate_peak(-data, x)
        estimate['offset'].set(value=-estimate['offset'].value,min=-estimate['offset'].max,max=-estimate['offset'].min)
        estimate['amplitude'].set(value=-estimate['amplitude'].value,min=-estimate['amplitude'].max,max=-estimate['amplitude'].min)
        return estimate

    @estimator('Peak (no offset)')
    def estimate_peak_no_offset(self, data, x):
        estimate = self.estimate_peak(data, x)
        estimate['offset'].set(value=0, min=-np.inf, max=np.inf, vary=False)
        return estimate

    @estimator('Dip (no offset)')
    def estimate_dip_no_offset(self, data, x):
        estimate = self.estimate_dip(data, x)
        estimate['offset'].set(value=0, min=-np.inf, max=np.inf, vary=False)
        return estimate
        
class ODMR5PeakSpectrum(FitModelBase):
    """ Five peak Lorentzian """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_param_hint('offset', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('center', value=1e-15, min=0, max=np.inf)
        self.set_param_hint('fwhm', value=0., min=0, max=np.inf)
        self.set_param_hint('hyperfine', value=2.16e6, min=-np.inf, max=np.inf)

    @staticmethod
    def _model_function(x, offset,amplitude,center,fwhm,hyperfine):
        xc0 = center - hyperfine*2
        xc1 = center - hyperfine
        xc2 = center
        xc3 = center + hyperfine
        xc4 = center + hyperfine*2
        #
        a0,a1,a2,a3,a4 = amplitude*amplitude_scaling
        #
        return func5Lorentz(x,offset,a0,xc0,fwhm, a1,xc1,fwhm, a2,xc2,fwhm, a3,xc3,fwhm, a4,xc4,fwhm)

    @estimator('Peak')
    def estimate_peak(self, data, x):
        data, x = sort_check_data(data, x)
        offset,amplitude,center,fwhm,[x_spacing,x_span,data_span] = single_peak_estimator(data, x)
        
        # correct since it was obtained from the area assuming a single peak
        fwhm = fwhm/3 # 1/3+2/3+1+2/3+1/3 = 3
        
        estimate = self.make_params()
        estimate['amplitude'].set(value=amplitude, min=0, max=100 * amplitude)
        estimate['fwhm'].set(value=fwhm, min=x_spacing/10, max=x_span)
        estimate['center'].set(value=center, min=min(x) - x_span / 2, max=max(x) + x_span / 2)
        estimate['offset'].set(value=offset, min=min(data) - data_span / 2, max=max(data) + data_span / 2 )
        return estimate
        
    @estimator('Dip')
    def estimate_dip(self, data, x):
        estimate = self.estimate_peak(-data, x)
        estimate['offset'].set(value=-estimate['offset'].value,min=-estimate['offset'].max,max=-estimate['offset'].min)
        estimate['amplitude'].set(value=-estimate['amplitude'].value,min=-estimate['amplitude'].max,max=-estimate['amplitude'].min)
        return estimate

    @estimator('Peak (no offset)')
    def estimate_peak_no_offset(self, data, x):
        estimate = self.estimate_peak(data, x)
        estimate['offset'].set(value=0, min=-np.inf, max=np.inf, vary=False)
        return estimate

    @estimator('Dip (no offset)')
    def estimate_dip_no_offset(self, data, x):
        estimate = self.estimate_dip(data, x)
        estimate['offset'].set(value=0, min=-np.inf, max=np.inf, vary=False)
        return estimate
        
class ODMR10PeakSpectrum(FitModelBase):
    """ Ten peak Lorentzian """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_param_hint('offset', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('center', value=1e-15, min=0, max=np.inf)
        self.set_param_hint('fwhm', value=0., min=0, max=np.inf)
        self.set_param_hint('hyperfine', value=2.16e6, min=-np.inf, max=np.inf)
        self.set_param_hint('mod_rate', value=4.5e6, min=-np.inf, max=np.inf)

    @staticmethod
    def _model_function(x, offset,amplitude,center,fwhm,hyperfine,mod_rate):
        xc0 = center - mod_rate - hyperfine*2
        xc1 = center - mod_rate - hyperfine
        xc2 = center - mod_rate
        xc3 = center - mod_rate + hyperfine
        xc4 = center - mod_rate + hyperfine*2
        xc5 = center + mod_rate - hyperfine*2
        xc6 = center + mod_rate - hyperfine
        xc7 = center + mod_rate
        xc8 = center + mod_rate + hyperfine
        xc9 = center + mod_rate + hyperfine*2
        #
        a0,a1,a2,a3,a4 = amplitude*amplitude_scaling
        a5,a6,a7,a8,a9 = amplitude*amplitude_scaling
        #
        return func10Lorentz(x,offset,a0,xc0,fwhm, a1,xc1,fwhm, a2,xc2,fwhm, a3,xc3,fwhm, a4,xc4,fwhm, a5,xc5,fwhm, a6,xc6,fwhm, a7,xc7,fwhm, a8,xc8,fwhm, a9,xc9,fwhm)

    @estimator('Peak')
    def estimate_peak(self, data, x):
        data, x = sort_check_data(data, x)
        offset,amplitude,center,fwhm,[x_spacing,x_span,data_span] = single_peak_estimator(data, x)
        
        # correct since it was obtained from the area assuming a single peak
        fwhm = fwhm/6 # ( 1/3+2/3+1+2/3+1/3 = 3 )*2 = 6
        
        estimate = self.make_params()
        estimate['amplitude'].set(value=amplitude, min=0, max=100 * amplitude)
        estimate['fwhm'].set(value=fwhm, min=x_spacing/10, max=x_span)
        estimate['center'].set(value=center, min=min(x) - x_span / 2, max=max(x) + x_span / 2)
        estimate['offset'].set(value=offset, min=min(data) - data_span / 2, max=max(data) + data_span / 2 )
        return estimate
        
    @estimator('Dip')
    def estimate_dip(self, data, x):
        estimate = self.estimate_peak(-data, x)
        estimate['offset'].set(value=-estimate['offset'].value,min=-estimate['offset'].max,max=-estimate['offset'].min)
        estimate['amplitude'].set(value=-estimate['amplitude'].value,min=-estimate['amplitude'].max,max=-estimate['amplitude'].min)
        return estimate

    @estimator('Peak (no offset)')
    def estimate_peak_no_offset(self, data, x):
        estimate = self.estimate_peak(data, x)
        estimate['offset'].set(value=0, min=-np.inf, max=np.inf, vary=False)
        return estimate

    @estimator('Dip (no offset)')
    def estimate_dip_no_offset(self, data, x):
        estimate = self.estimate_dip(data, x)
        estimate['offset'].set(value=0, min=-np.inf, max=np.inf, vary=False)
        return estimate

########################################################################
########################################################################

class ODMR1DispersionSpectrum(FitModelBase):
    """ Single peak Lorentzian derivative """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_param_hint('offset', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('center', value=1e-15, min=0, max=np.inf)
        self.set_param_hint('width', value=0., min=0, max=np.inf)

    @staticmethod
    def _model_function(x, offset,amplitude,center,width):
        return funcDerivativeLorentz(x,offset,amplitude,center,width)

    @estimator('Peak')
    def estimate_peak(self, data, x):
        data, x = sort_check_data(data, x)
        offset,amplitude,center,width,[x_spacing,x_span,data_span] = single_dispersion_estimator(data, x)
        
        estimate = self.make_params()
        estimate['amplitude'].set(value=amplitude, min=0, max=100 * amplitude)
        estimate['width'].set(value=width, min=x_spacing/10, max=x_span)
        estimate['center'].set(value=center, min=min(x) - x_span / 2, max=max(x) + x_span / 2)
        estimate['offset'].set(value=offset, min=min(data) - data_span / 2, max=max(data) + data_span / 2 )
        return estimate
        
    @estimator('Dip')
    def estimate_dip(self, data, x):
        estimate = self.estimate_peak(-data, x)
        estimate['offset'].set(value=-estimate['offset'].value,min=-estimate['offset'].max,max=-estimate['offset'].min)
        estimate['amplitude'].set(value=-estimate['amplitude'].value,min=-estimate['amplitude'].max,max=-estimate['amplitude'].min)
        return estimate

    @estimator('Peak (no offset)')
    def estimate_peak_no_offset(self, data, x):
        estimate = self.estimate_peak(data, x)
        estimate['offset'].set(value=0, min=-np.inf, max=np.inf, vary=False)
        return estimate

    @estimator('Dip (no offset)')
    def estimate_dip_no_offset(self, data, x):
        estimate = self.estimate_dip(data, x)
        estimate['offset'].set(value=0, min=-np.inf, max=np.inf, vary=False)
        return estimate
        
class ODMR3DispersionSpectrum(FitModelBase):
    """ Three peak Lorentzian derivative """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_param_hint('offset', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('center', value=1e-15, min=0, max=np.inf)
        self.set_param_hint('width', value=0., min=0, max=np.inf)
        self.set_param_hint('hyperfine', value=2.16e6, min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude_factor', value=1, min=0.5, max=1.5)

    @staticmethod
    def _model_function(x, offset,amplitude,center,width,hyperfine,amplitude_factor):
        xc0 = center - hyperfine
        xc1 = center
        xc2 = center + hyperfine
        return func3DerivativeLorentz(x,offset,amplitude*amplitude_factor,xc0,width, amplitude,xc1,width, amplitude*amplitude_factor,xc2,width)
    
    @estimator('Peak (center self correl)')
    def estimate_peak_selfcorr(self, data, x):
        data, x = sort_check_data(data, x)
        offset,amplitude,center,width,[x_spacing,x_span,data_span] = single_dispersion_estimator(data, x, useMinMaxCenter=False)
        
        estimate = self.make_params()
        estimate['amplitude'].set(value=amplitude, min=0, max=100 * amplitude)
        estimate['width'].set(value=width, min=x_spacing/10, max=x_span)
        estimate['center'].set(value=center, min=min(x) - x_span / 2, max=max(x) + x_span / 2)
        estimate['offset'].set(value=offset, min=min(data) - data_span / 2, max=max(data) + data_span / 2 )
        estimate['amplitude_factor'].set(value=1, min=0.5, max=1.5)
        return estimate
        
    @estimator('Peak')
    def estimate_peak(self, data, x):
        data, x = sort_check_data(data, x)
        offset,amplitude,center,width,[x_spacing,x_span,data_span] = single_dispersion_estimator(data, x)
        
        estimate = self.make_params()
        estimate['amplitude'].set(value=amplitude, min=0, max=100 * amplitude)
        estimate['width'].set(value=width, min=x_spacing/10, max=x_span)
        estimate['center'].set(value=center, min=min(x) - x_span / 2, max=max(x) + x_span / 2)
        estimate['offset'].set(value=offset, min=min(data) - data_span / 2, max=max(data) + data_span / 2 )
        estimate['amplitude_factor'].set(value=1, min=0.5, max=1.5)
        return estimate
        
    @estimator('Dip')
    def estimate_dip(self, data, x):
        estimate = self.estimate_peak(-data, x)
        estimate['offset'].set(value=-estimate['offset'].value,min=-estimate['offset'].max,max=-estimate['offset'].min)
        estimate['amplitude'].set(value=-estimate['amplitude'].value,min=-estimate['amplitude'].max,max=-estimate['amplitude'].min)
        estimate['amplitude_factor'].set(value=1, min=0.5, max=1.5)
        return estimate

    @estimator('Peak (no offset)')
    def estimate_peak_no_offset(self, data, x):
        estimate = self.estimate_peak(data, x)
        estimate['offset'].set(value=0, min=-np.inf, max=np.inf, vary=False)
        return estimate

    @estimator('Dip (no offset)')
    def estimate_dip_no_offset(self, data, x):
        estimate = self.estimate_dip(data, x)
        estimate['offset'].set(value=0, min=-np.inf, max=np.inf, vary=False)
        return estimate
        
class ODMR6DispersionSpectrum(FitModelBase):
    """ Six peak Lorentzian derivative """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_param_hint('offset', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('center', value=1e-15, min=0, max=np.inf)
        self.set_param_hint('width', value=0., min=0, max=np.inf)
        self.set_param_hint('hyperfine', value=2.16e6, min=-np.inf, max=np.inf)
        self.set_param_hint('mod_rate', value=4.5e6, min=-np.inf, max=np.inf)

    @staticmethod
    def _model_function(x, offset,amplitude,center,width,hyperfine,mod_rate):
        xc0 = center - mod_rate - hyperfine
        xc1 = center - mod_rate
        xc2 = center - mod_rate + hyperfine
        xc3 = center + mod_rate - hyperfine
        xc4 = center + mod_rate
        xc5 = center + mod_rate + hyperfine
        return func6DerivativeLorentz(x,offset,amplitude,xc0,width, amplitude,xc1,width, amplitude,xc2,width, amplitude,xc3,width, amplitude,xc4,width, amplitude,xc5,width)
    
    @estimator('Peak')
    def estimate_peak(self, data, x):
        data, x = sort_check_data(data, x)
        offset,amplitude,center,width,[x_spacing,x_span,data_span] = single_dispersion_estimator(data, x)
        
        estimate = self.make_params()
        estimate['amplitude'].set(value=amplitude, min=0, max=100 * amplitude)
        estimate['width'].set(value=width, min=x_spacing/10, max=x_span)
        estimate['center'].set(value=center, min=min(x) - x_span / 2, max=max(x) + x_span / 2)
        estimate['offset'].set(value=offset, min=min(data) - data_span / 2, max=max(data) + data_span / 2 )
        return estimate
        
    @estimator('Dip')
    def estimate_dip(self, data, x):
        estimate = self.estimate_peak(-data, x)
        estimate['offset'].set(value=-estimate['offset'].value,min=-estimate['offset'].max,max=-estimate['offset'].min)
        estimate['amplitude'].set(value=-estimate['amplitude'].value,min=-estimate['amplitude'].max,max=-estimate['amplitude'].min)
        return estimate

    @estimator('Peak (no offset)')
    def estimate_peak_no_offset(self, data, x):
        estimate = self.estimate_peak(data, x)
        estimate['offset'].set(value=0, min=-np.inf, max=np.inf, vary=False)
        return estimate

    @estimator('Dip (no offset)')
    def estimate_dip_no_offset(self, data, x):
        estimate = self.estimate_dip(data, x)
        estimate['offset'].set(value=0, min=-np.inf, max=np.inf, vary=False)
        return estimate
        
class ODMR5DispersionSpectrum(FitModelBase):
    """ Five peak Lorentzian derivative """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_param_hint('offset', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('center', value=1e-15, min=0, max=np.inf)
        self.set_param_hint('width', value=0., min=0, max=np.inf)
        self.set_param_hint('hyperfine', value=2.16e6, min=-np.inf, max=np.inf)
        # self.set_param_hint('amplitude_factor_1', value=2/3, min=0, max=np.inf)
        # self.set_param_hint('amplitude_factor_2', value=1/3, min=0, max=np.inf)

    @staticmethod
    def _model_function(x, offset,amplitude,center,width,hyperfine):
        xc0 = center - hyperfine*2
        xc1 = center - hyperfine
        xc2 = center
        xc3 = center + hyperfine
        xc4 = center + hyperfine*2
        #
        a0,a1,a2,a3,a4 = amplitude*amplitude_scaling
        # to do: add amplitude_scaling as a free parameter:
        # a2 = amplitude
        # a1,a3 = amplitude*amplitude_factor_1,amplitude*amplitude_factor_1
        # a0,a4 = amplitude*amplitude_factor_2,amplitude*amplitude_factor_2
        #
        return func5DerivativeLorentz(x,offset,a0,xc0,width, a1,xc1,width, a2,xc2,width, a3,xc3,width, a4,xc4,width)
    
    @estimator('Peak')
    def estimate_peak(self, data, x):
        data, x = sort_check_data(data, x)
        offset,amplitude,center,width,[x_spacing,x_span,data_span] = single_dispersion_estimator(data, x)
        
        estimate = self.make_params()
        estimate['amplitude'].set(value=amplitude, min=0, max=100 * amplitude)
        estimate['width'].set(value=width, min=x_spacing/10, max=x_span)
        estimate['center'].set(value=center, min=min(x) - x_span / 2, max=max(x) + x_span / 2)
        estimate['offset'].set(value=offset, min=min(data) - data_span / 2, max=max(data) + data_span / 2 )
        return estimate
        
    @estimator('Dip')
    def estimate_dip(self, data, x):
        estimate = self.estimate_peak(-data, x)
        estimate['offset'].set(value=-estimate['offset'].value,min=-estimate['offset'].max,max=-estimate['offset'].min)
        estimate['amplitude'].set(value=-estimate['amplitude'].value,min=-estimate['amplitude'].max,max=-estimate['amplitude'].min)
        return estimate

    @estimator('Peak (no offset)')
    def estimate_peak_no_offset(self, data, x):
        estimate = self.estimate_peak(data, x)
        estimate['offset'].set(value=0, min=-np.inf, max=np.inf, vary=False)
        return estimate

    @estimator('Dip (no offset)')
    def estimate_dip_no_offset(self, data, x):
        estimate = self.estimate_dip(data, x)
        estimate['offset'].set(value=0, min=-np.inf, max=np.inf, vary=False)
        return estimate
    
    @estimator('Peak (center self correl)')
    def estimate_peak_selfcorr(self, data, x):
        data, x = sort_check_data(data, x)
        offset,amplitude,center,width,[x_spacing,x_span,data_span] = single_dispersion_estimator(data, x, useMinMaxCenter=False)
        
        estimate = self.make_params()
        estimate['amplitude'].set(value=amplitude, min=0, max=100 * amplitude)
        estimate['width'].set(value=width, min=x_spacing/10, max=x_span)
        estimate['center'].set(value=center, min=min(x) - x_span / 2, max=max(x) + x_span / 2)
        estimate['offset'].set(value=offset, min=min(data) - data_span / 2, max=max(data) + data_span / 2 )
        return estimate
        
    @estimator('Dip (center self correl)')
    def estimate_dip_selfcorr(self, data, x):
        estimate = self.estimate_peak_selfcorr(-data, x)
        estimate['offset'].set(value=-estimate['offset'].value,min=-estimate['offset'].max,max=-estimate['offset'].min)
        estimate['amplitude'].set(value=-estimate['amplitude'].value,min=-estimate['amplitude'].max,max=-estimate['amplitude'].min)
        return estimate

        
class ODMR10DispersionSpectrum(FitModelBase):
    """ Ten peak Lorentzian derivative """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_param_hint('offset', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('center', value=1e-15, min=0, max=np.inf)
        self.set_param_hint('width', value=0., min=0, max=np.inf)
        self.set_param_hint('hyperfine', value=2.16e6, min=-np.inf, max=np.inf)
        self.set_param_hint('mod_rate', value=4.5e6, min=-np.inf, max=np.inf)

    @staticmethod
    def _model_function(x, offset,amplitude,center,width,hyperfine,mod_rate):
        xc0 = center - mod_rate - hyperfine*2
        xc1 = center - mod_rate - hyperfine
        xc2 = center - mod_rate
        xc3 = center - mod_rate + hyperfine
        xc4 = center - mod_rate + hyperfine*2
        xc5 = center + mod_rate - hyperfine*2
        xc6 = center + mod_rate - hyperfine
        xc7 = center + mod_rate
        xc8 = center + mod_rate + hyperfine
        xc9 = center + mod_rate + hyperfine*2
        #
        a0,a1,a2,a3,a4 = amplitude*amplitude_scaling
        a5,a6,a7,a8,a9 = amplitude*amplitude_scaling
        #
        return func10DerivativeLorentz(x,offset,a0,xc0,width, a1,xc1,width, a2,xc2,width, a3,xc3,width, a4,xc4,width, a5,xc5,width, a6,xc6,width, a7,xc7,width, a8,xc8,width, a9,xc9,width)

    @estimator('Peak')
    def estimate_peak(self, data, x):
        data, x = sort_check_data(data, x)
        offset,amplitude,center,width,[x_spacing,x_span,data_span] = single_dispersion_estimator(data, x)
        
        estimate = self.make_params()
        estimate['amplitude'].set(value=amplitude, min=0, max=100 * amplitude)
        estimate['width'].set(value=width, min=x_spacing/10, max=x_span)
        estimate['center'].set(value=center, min=min(x) - x_span / 2, max=max(x) + x_span / 2)
        estimate['offset'].set(value=offset, min=min(data) - data_span / 2, max=max(data) + data_span / 2 )
        return estimate
        
    @estimator('Dip')
    def estimate_dip(self, data, x):
        estimate = self.estimate_peak(-data, x)
        estimate['offset'].set(value=-estimate['offset'].value,min=-estimate['offset'].max,max=-estimate['offset'].min)
        estimate['amplitude'].set(value=-estimate['amplitude'].value,min=-estimate['amplitude'].max,max=-estimate['amplitude'].min)
        return estimate

    @estimator('Peak (no offset)')
    def estimate_peak_no_offset(self, data, x):
        estimate = self.estimate_peak(data, x)
        estimate['offset'].set(value=0, min=-np.inf, max=np.inf, vary=False)
        return estimate

    @estimator('Dip (no offset)')
    def estimate_dip_no_offset(self, data, x):
        estimate = self.estimate_dip(data, x)
        estimate['offset'].set(value=0, min=-np.inf, max=np.inf, vary=False)
        return estimate

   
##
## New model class to allow IQ simultaneous fit
##
class ODMR5DispersionSpectrum_IQ(FitModelBase):
    """
    Five peak Lorentzian derivative for complex data.
    Using amplitude_factor for left and right peaks:
    amplitude: amplitude of central peak
    amplitude*amplitude_factor_1: amplitude of secondary peaks (ideally 2/3 of central peak)
    amplitude*amplitude_factor_2: amplitude of tertiary peaks (ideally 1/3 of central peak)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_param_hint('offset', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('center', value=1e-15, min=0, max=np.inf)
        self.set_param_hint('width', value=0., min=0, max=np.inf)
        self.set_param_hint('hyperfine', value=2.16e6, min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude_factor_1', value=2/3, min=0, max=np.inf)
        self.set_param_hint('amplitude_factor_2', value=1/3, min=0, max=np.inf)
        self.set_param_hint('IQphase', value=0, min=-np.pi, max=np.pi)

    @staticmethod
    def _model_function(x, offset,amplitude,center,width,hyperfine,
                        amplitude_factor_1,amplitude_factor_2,IQphase):
        xc0 = center - hyperfine*2
        xc1 = center - hyperfine
        xc2 = center
        xc3 = center + hyperfine
        xc4 = center + hyperfine*2
        #
        ##a0,a1,a2,a3,a4 = amplitude*amplitude_scaling
        #a2 = amplitude
        #a1,a3 = amplitude*amplitude_factor_1,amplitude*amplitude_factor_1
        #a0,a4 = amplitude*amplitude_factor_2,amplitude*amplitude_factor_2
        #
        #a0I,a1I,a2I,a3I,a4I = amplitude*amplitude_scaling*np.cos(IQphase)
        #a0Q,a1Q,a2Q,a3Q,a4Q = amplitude*amplitude_scaling*np.sin(IQphase)
        a2I = amplitude*np.cos(IQphase)
        a1I,a3I = a2I*amplitude_factor_1,a2I*amplitude_factor_1
        a0I,a4I = a2I*amplitude_factor_2,a2I*amplitude_factor_2
        a2Q = amplitude*np.sin(IQphase)
        a1Q,a3Q = a2Q*amplitude_factor_1,a2Q*amplitude_factor_1
        a0Q,a4Q = a2Q*amplitude_factor_2,a2Q*amplitude_factor_2
        
        real = func5DerivativeLorentz(x,offset,a0I,xc0,width, a1I,xc1,width, a2I,xc2,width, a3I,xc3,width, a4I,xc4,width)
        imag = func5DerivativeLorentz(x,offset,a0Q,xc0,width, a1Q,xc1,width, a2Q,xc2,width, a3Q,xc3,width, a4Q,xc4,width)
        return (real + 1j*imag).view()
    
    @estimator('Peak')
    def estimate_peak(self, data, x):
        data, x = sort_check_data(data, x)
        offsetI,amplitudeI,centerI,widthI,[x_spacing,x_span,data_spanI] = single_dispersion_estimator(data.real, x)
        offsetQ,amplitudeQ,centerQ,widthQ,[x_spacing,x_span,data_spanQ] = single_dispersion_estimator(data.imag, x)
        data_span = max(data_spanI,data_spanQ)
        
        amplitude = np.sqrt(amplitudeI**2+amplitudeQ**2)
        IQphase = np.arctan2(amplitudeQ,amplitudeI)
        #
        center = centerI if amplitudeI>amplitudeQ else centerQ
        width = widthI if amplitudeI>amplitudeQ else widthQ
        offset = (offsetI+offsetQ)/2
        
        estimate = self.make_params()
        estimate['amplitude'].set(value=amplitude, min=0, max=100 * amplitude)
        estimate['width'].set(value=width, min=x_spacing/10, max=x_span)
        estimate['center'].set(value=center, min=min(x) - x_span / 2, max=max(x) + x_span / 2)
        estimate['offset'].set(value=offset, min=min(min(data.real),min(data.imag)) - data_span / 2, max=max(max(data.real),max(data.imag)) + data_span / 2 )
        estimate['amplitude_factor_1'].set(value=2/3)
        estimate['amplitude_factor_2'].set(value=1/3)
        estimate['IQphase'].set(value=IQphase,min=-np.pi, max=np.pi)
        return estimate
        
    @estimator('Peak (center self correl)')
    def estimate_peak_selfcorr(self, data, x):
        data, x = sort_check_data(data, x)
        offsetI,amplitudeI,centerI,widthI,[x_spacing,x_span,data_spanI] = single_dispersion_estimator(data.real, x,useMinMaxCenter=False)
        offsetQ,amplitudeQ,centerQ,widthQ,[x_spacing,x_span,data_spanQ] = single_dispersion_estimator(data.imag, x,useMinMaxCenter=False)
        data_span = max(data_spanI,data_spanQ)
        
        amplitude = np.sqrt(amplitudeI**2+amplitudeQ**2)
        IQphase = np.arctan2(amplitudeQ,amplitudeI)
        #
        center = centerI if amplitudeI>amplitudeQ else centerQ
        width = widthI if amplitudeI>amplitudeQ else widthQ
        offset = (offsetI+offsetQ)/2
        
        estimate = self.make_params()
        estimate['amplitude'].set(value=amplitude, min=0, max=100 * amplitude)
        estimate['width'].set(value=width, min=x_spacing/10, max=x_span)
        estimate['center'].set(value=center, min=min(x) - x_span / 2, max=max(x) + x_span / 2)
        estimate['offset'].set(value=offset, min=min(min(data.real),min(data.imag)) - data_span / 2, max=max(max(data.real),max(data.imag)) + data_span / 2 )
        estimate['amplitude_factor_1'].set(value=2/3)
        estimate['amplitude_factor_2'].set(value=1/3)
        estimate['IQphase'].set(value=IQphase,min=-np.pi, max=np.pi)
        return estimate
    
class ODMR7DispersionSpectrum_IQ(FitModelBase):
    """ 
    Seven peak Lorentzian derivative for complex data.
    Using amplitude_factor for left and right peaks:
    amplitude: amplitude of central peak
    amplitude*amplitude_factor_1: amplitude of secondary peaks (ideally 2/3 of central peak)
    amplitude*amplitude_factor_2: amplitude of tertiary peaks (ideally 1/3 of central peak)
    amplitude*amplitude_factor_3: amplitude of extra peaks (harmonics, ideally equal to 0)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_param_hint('offset', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('center', value=1e-15, min=0, max=np.inf)
        self.set_param_hint('width', value=0., min=0, max=np.inf)
        self.set_param_hint('hyperfine', value=2.16e6, min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude_factor_1', value=2/3, min=0, max=np.inf)
        self.set_param_hint('amplitude_factor_2', value=1/3, min=0, max=np.inf)
        self.set_param_hint('amplitude_factor_3', value=1/9, min=0, max=np.inf)
        self.set_param_hint('IQphase', value=0, min=-np.pi, max=np.pi)

    @staticmethod
    def _model_function(x, offset,amplitude,center,width,hyperfine,
                        amplitude_factor_1,amplitude_factor_2,amplitude_factor_3,IQphase):
        xc0 = center - hyperfine*3
        xc1 = center - hyperfine*2
        xc2 = center - hyperfine
        xc3 = center
        xc4 = center + hyperfine
        xc5 = center + hyperfine*2
        xc6 = center + hyperfine*3
        #
        #a3 = amplitude
        #a2,a4 = amplitude*amplitude_factor_1,amplitude*amplitude_factor_1
        #a1,a5 = amplitude*amplitude_factor_2,amplitude*amplitude_factor_2
        #a0,a6 = amplitude*amplitude_factor_3,amplitude*amplitude_factor_3
        #
        #a0I,a1I,a2I,a3I,a4I = amplitude*amplitude_scaling*np.cos(IQphase)
        #a0Q,a1Q,a2Q,a3Q,a4Q = amplitude*amplitude_scaling*np.sin(IQphase)
        a3I = amplitude*np.cos(IQphase)
        a2I,a4I = a3I*amplitude_factor_1,a3I*amplitude_factor_1
        a1I,a5I = a3I*amplitude_factor_2,a3I*amplitude_factor_2
        a0I,a6I = a3I*amplitude_factor_3,a3I*amplitude_factor_3
        a3Q = amplitude*np.sin(IQphase)
        a2Q,a4Q = a3Q*amplitude_factor_1,a3Q*amplitude_factor_1
        a1Q,a5Q = a3Q*amplitude_factor_2,a3Q*amplitude_factor_2
        a0Q,a6Q = a3Q*amplitude_factor_3,a3Q*amplitude_factor_3
        
        real = func7DerivativeLorentz(x,offset,a0I,xc0,width, a1I,xc1,width, a2I,xc2,width, a3I,xc3,width, a4I,xc4,width, a5I,xc5,width, a6I,xc6,width)
        imag = func7DerivativeLorentz(x,offset,a0Q,xc0,width, a1Q,xc1,width, a2Q,xc2,width, a3Q,xc3,width, a4Q,xc4,width, a5Q,xc5,width, a6Q,xc6,width)
        return (real + 1j*imag).view()
    
    @estimator('Peak')
    def estimate_peak(self, data, x):
        data, x = sort_check_data(data, x)
        offsetI,amplitudeI,centerI,widthI,[x_spacing,x_span,data_spanI] = single_dispersion_estimator(data.real, x)
        offsetQ,amplitudeQ,centerQ,widthQ,[x_spacing,x_span,data_spanQ] = single_dispersion_estimator(data.imag, x)
        data_span = max(data_spanI,data_spanQ)
        
        amplitude = np.sqrt(amplitudeI**2+amplitudeQ**2)
        IQphase = np.arctan2(amplitudeQ,amplitudeI)
        #
        center = centerI if amplitudeI>amplitudeQ else centerQ
        width = widthI if amplitudeI>amplitudeQ else widthQ
        offset = (offsetI+offsetQ)/2
        
        estimate = self.make_params()
        estimate['amplitude'].set(value=amplitude, min=0, max=100 * amplitude)
        estimate['width'].set(value=width, min=x_spacing/10, max=x_span)
        estimate['center'].set(value=center, min=min(x) - x_span / 2, max=max(x) + x_span / 2)
        estimate['offset'].set(value=offset, min=min(min(data.real),min(data.imag)) - data_span / 2, max=max(max(data.real),max(data.imag)) + data_span / 2 )
        estimate['amplitude_factor_1'].set(value=2/3)
        estimate['amplitude_factor_2'].set(value=1/3)
        estimate['amplitude_factor_3'].set(value=1/9)
        estimate['IQphase'].set(value=IQphase,min=-np.pi, max=np.pi)
        return estimate
        
    @estimator('Peak (center self correl)')
    def estimate_peak_selfcorr(self, data, x):
        data, x = sort_check_data(data, x)
        offsetI,amplitudeI,centerI,widthI,[x_spacing,x_span,data_spanI] = single_dispersion_estimator(data.real, x,useMinMaxCenter=False)
        offsetQ,amplitudeQ,centerQ,widthQ,[x_spacing,x_span,data_spanQ] = single_dispersion_estimator(data.imag, x,useMinMaxCenter=False)
        data_span = max(data_spanI,data_spanQ)
        
        amplitude = np.sqrt(amplitudeI**2+amplitudeQ**2)
        IQphase = np.arctan2(amplitudeQ,amplitudeI)
        #
        center = centerI if amplitudeI>amplitudeQ else centerQ
        width = widthI if amplitudeI>amplitudeQ else widthQ
        offset = (offsetI+offsetQ)/2
        
        estimate = self.make_params()
        estimate['amplitude'].set(value=amplitude, min=0, max=100 * amplitude)
        estimate['width'].set(value=width, min=x_spacing/10, max=x_span)
        estimate['center'].set(value=center, min=min(x) - x_span / 2, max=max(x) + x_span / 2)
        estimate['offset'].set(value=offset, min=min(min(data.real),min(data.imag)) - data_span / 2, max=max(max(data.real),max(data.imag)) + data_span / 2 )
        estimate['amplitude_factor_1'].set(value=2/3)
        estimate['amplitude_factor_2'].set(value=1/3)
        estimate['amplitude_factor_3'].set(value=1/9)
        estimate['IQphase'].set(value=IQphase,min=-np.pi, max=np.pi)
        return estimate

        
class ODMR3DispersionSpectrum_IQ(FitModelBase):
    """
    Three peak Lorentzian derivative for complex data
    Using amplitude_factor for left and right peaks:
    amplitude: amplitude of central peak
    amplitude*amplitude_factor: amplitude of left peak and right peak
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_param_hint('offset', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('center', value=1e-15, min=0, max=np.inf)
        self.set_param_hint('width', value=0., min=0, max=np.inf)
        self.set_param_hint('hyperfine', value=2.16e6, min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude_factor', value=1, min=0.5, max=1.5)
        self.set_param_hint('IQphase', value=0, min=-np.pi, max=np.pi)

    @staticmethod
    def _model_function(x, offset,amplitude,center,width,hyperfine,
                        amplitude_factor,IQphase):
        xc0 = center - hyperfine
        xc1 = center
        xc2 = center + hyperfine
        #
        a1I = amplitude*np.cos(IQphase)
        a0I = a1I*amplitude_factor
        a2I = a1I*amplitude_factor
        a1Q = amplitude*np.sin(IQphase)
        a0Q = a1Q*amplitude_factor
        a2Q = a1Q*amplitude_factor
        
        real = func3DerivativeLorentz(x,offset,a0I,xc0,width, a1I,xc1,width, a2I,xc2,width)
        imag = func3DerivativeLorentz(x,offset,a0Q,xc0,width, a1Q,xc1,width, a2Q,xc2,width)
        return (real + 1j*imag).view()
    
    @estimator('Peak')
    def estimate_peak(self, data, x):
        data, x = sort_check_data(data, x)
        offsetI,amplitudeI,centerI,widthI,[x_spacing,x_span,data_spanI] = single_dispersion_estimator(data.real, x)
        offsetQ,amplitudeQ,centerQ,widthQ,[x_spacing,x_span,data_spanQ] = single_dispersion_estimator(data.imag, x)
        data_span = max(data_spanI,data_spanQ)
        
        amplitude = np.sqrt(amplitudeI**2+amplitudeQ**2)
        IQphase = np.arctan2(amplitudeQ,amplitudeI)
        #
        center = centerI if amplitudeI>amplitudeQ else centerQ
        width = widthI if amplitudeI>amplitudeQ else widthQ
        offset = (offsetI+offsetQ)/2
        
        estimate = self.make_params()
        estimate['amplitude'].set(value=amplitude, min=0, max=100 * amplitude)
        estimate['width'].set(value=width, min=x_spacing/10, max=x_span)
        estimate['center'].set(value=center, min=min(x) - x_span / 2, max=max(x) + x_span / 2)
        estimate['offset'].set(value=offset, min=min(min(data.real),min(data.imag)) - data_span / 2, max=max(max(data.real),max(data.imag)) + data_span / 2 )
        estimate['amplitude_factor'].set(value=1, min=0.5, max=1.5)
        estimate['IQphase'].set(value=IQphase,min=-np.pi, max=np.pi)
        return estimate
        
    @estimator('Peak (center self correl)')
    def estimate_peak_selfcorr(self, data, x):
        data, x = sort_check_data(data, x)
        offsetI,amplitudeI,centerI,widthI,[x_spacing,x_span,data_spanI] = single_dispersion_estimator(data.real, x,useMinMaxCenter=False)
        offsetQ,amplitudeQ,centerQ,widthQ,[x_spacing,x_span,data_spanQ] = single_dispersion_estimator(data.imag, x,useMinMaxCenter=False)
        data_span = max(data_spanI,data_spanQ)
        
        amplitude = np.sqrt(amplitudeI**2+amplitudeQ**2)
        IQphase = np.arctan2(amplitudeQ,amplitudeI)
        #
        center = centerI if amplitudeI>amplitudeQ else centerQ
        width = widthI if amplitudeI>amplitudeQ else widthQ
        offset = (offsetI+offsetQ)/2
        
        estimate = self.make_params()
        estimate['amplitude'].set(value=amplitude, min=0, max=100 * amplitude)
        estimate['width'].set(value=width, min=x_spacing/10, max=x_span)
        estimate['center'].set(value=center, min=min(x) - x_span / 2, max=max(x) + x_span / 2)
        estimate['offset'].set(value=offset, min=min(min(data.real),min(data.imag)) - data_span / 2, max=max(max(data.real),max(data.imag)) + data_span / 2 )
        estimate['amplitude_factor'].set(value=1, min=0.5, max=1.5)
        estimate['IQphase'].set(value=IQphase,min=-np.pi, max=np.pi)
        return estimate
    
class ODMR7DispersionSpectrum(FitModelBase):
    """ 
    Seven peak Lorentzian derivative for complex data.
    Using amplitude_factor for left and right peaks:
    amplitude: amplitude of central peak
    amplitude*amplitude_factor_1: amplitude of secondary peaks (ideally 2/3 of central peak)
    amplitude*amplitude_factor_2: amplitude of tertiary peaks (ideally 1/3 of central peak)
    amplitude*amplitude_factor_3: amplitude of extra peaks (harmonics, ideally equal to 0)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_param_hint('offset', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('center', value=1e-15, min=0, max=np.inf)
        self.set_param_hint('width', value=0., min=0, max=np.inf)
        self.set_param_hint('hyperfine', value=2.16e6, min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude_factor_1', value=2/3, min=0, max=np.inf)
        self.set_param_hint('amplitude_factor_2', value=1/3, min=0, max=np.inf)
        self.set_param_hint('amplitude_factor_3', value=1/9, min=0, max=np.inf)

    @staticmethod
    def _model_function(x, offset,amplitude,center,width,hyperfine,
                        amplitude_factor_1,amplitude_factor_2,amplitude_factor_3):
        xc0 = center - hyperfine*3
        xc1 = center - hyperfine*2
        xc2 = center - hyperfine
        xc3 = center
        xc4 = center + hyperfine
        xc5 = center + hyperfine*2
        xc6 = center + hyperfine*3
        #
        #a3 = amplitude
        #a2,a4 = amplitude*amplitude_factor_1,amplitude*amplitude_factor_1
        #a1,a5 = amplitude*amplitude_factor_2,amplitude*amplitude_factor_2
        #a0,a6 = amplitude*amplitude_factor_3,amplitude*amplitude_factor_3
        #
        #a0I,a1I,a2I,a3I,a4I = amplitude*amplitude_scaling*np.cos(IQphase)
        #a0Q,a1Q,a2Q,a3Q,a4Q = amplitude*amplitude_scaling*np.sin(IQphase)
        a3I = amplitude
        a2I,a4I = a3I*amplitude_factor_1,a3I*amplitude_factor_1
        a1I,a5I = a3I*amplitude_factor_2,a3I*amplitude_factor_2
        a0I,a6I = a3I*amplitude_factor_3,a3I*amplitude_factor_3
        
        real = func7DerivativeLorentz(x,offset,a0I,xc0,width, a1I,xc1,width, a2I,xc2,width, a3I,xc3,width, a4I,xc4,width, a5I,xc5,width, a6I,xc6,width)
        return real
    
    @estimator('Peak')
    def estimate_peak(self, data, x):
        data, x = sort_check_data(data, x)
        offsetI,amplitudeI,centerI,widthI,[x_spacing,x_span,data_span] = single_dispersion_estimator(data.real, x)

        amplitude = amplitudeI
        #
        center = centerI
        width = widthI
        offset = offsetI
        
        estimate = self.make_params()
        estimate['amplitude'].set(value=amplitude, min=0, max=100 * amplitude)
        estimate['width'].set(value=width, min=x_spacing/10, max=x_span)
        estimate['center'].set(value=center, min=min(x) - x_span / 2, max=max(x) + x_span / 2)
        estimate['offset'].set(value=offset, min=min(data.real) - data_span / 2, max=max(data.real) + data_span / 2 )
        estimate['amplitude_factor_1'].set(value=2/3)
        estimate['amplitude_factor_2'].set(value=1/3)
        estimate['amplitude_factor_3'].set(value=1/9)
        return estimate
        
    @estimator('Peak (center self correl)')
    def estimate_peak_selfcorr(self, data, x):
        data, x = sort_check_data(data, x)
        offsetI,amplitudeI,centerI,widthI,[x_spacing,x_span,data_span] = single_dispersion_estimator(data.real, x,useMinMaxCenter=False)

        amplitude = amplitudeI
        #
        center = centerI
        width = widthI
        offset = offsetI
        
        estimate = self.make_params()
        estimate['amplitude'].set(value=amplitude, min=0, max=100 * amplitude)
        estimate['width'].set(value=width, min=x_spacing/10, max=x_span)
        estimate['center'].set(value=center, min=min(x) - x_span / 2, max=max(x) + x_span / 2)
        estimate['offset'].set(value=offset, min=min(data.real) - data_span / 2, max=max(data.real) + data_span / 2 )
        estimate['amplitude_factor_1'].set(value=2/3)
        estimate['amplitude_factor_2'].set(value=1/3)
        estimate['amplitude_factor_3'].set(value=1/9)
        return estimate
        
        
class ODMR3PeakSpectrum_IQ(FitModelBase):
    """
    Three peak Lorentzian for complex data
    Using amplitude_factor for left and right peaks:
    amplitude: amplitude of central peak
    amplitude*amplitude_factor: amplitude of left peak and right peak
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_param_hint('offset', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('center', value=1e-15, min=0, max=np.inf)
        self.set_param_hint('fwhm', value=0., min=0, max=np.inf)
        self.set_param_hint('hyperfine', value=2.16e6, min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude_factor', value=1, min=0, max=np.inf)
        self.set_param_hint('IQphase', value=0, min=-np.pi, max=np.pi)

    @staticmethod
    def _model_function(x, offset,amplitude,center,fwhm,hyperfine,amplitude_factor,IQphase):
        xc0 = center - hyperfine
        xc1 = center
        xc2 = center + hyperfine
        
        a1I = amplitude*np.cos(IQphase)
        a0I = a1I*amplitude_factor
        a2I = a1I*amplitude_factor
        a1Q = amplitude*np.sin(IQphase)
        a0Q = a1Q*amplitude_factor
        a2Q = a1Q*amplitude_factor
        
        real = func3Lorentz(x,offset,a0I,xc0,fwhm, a1I,xc1,fwhm, a2I,xc2,fwhm)
        imag = func3Lorentz(x,offset,a0Q,xc0,fwhm, a1Q,xc1,fwhm, a2Q,xc2,fwhm)
        return (real + 1j*imag).view()
    
    @estimator('Peak (center self correl)')
    def estimate_peak_selfcorr(self, data, x):
        data, x = sort_check_data(data, x)
        offsetI,amplitudeI,centerI,fwhmI,[x_spacing,x_span,data_spanI] = single_peak_estimator(data.real, x, useMinMaxCenter=False)
        offsetQ,amplitudeQ,centerQ,fwhmQ,[x_spacing,x_span,data_spanQ] = single_peak_estimator(data.imag, x, useMinMaxCenter=False)
        data_span = max(data_spanI,data_spanQ)
        
        amplitude = np.sqrt(amplitudeI**2+amplitudeQ**2)
        IQphase = np.arctan2(amplitudeQ,amplitudeI)
        #
        center = centerI if amplitudeI>amplitudeQ else centerQ
        fwhm = fwhmI if amplitudeI>amplitudeQ else fwhmQ
        offset = (offsetI+offsetQ)/2
        
        # correct since it was obtained from the area assuming a single peak
        fwhm = fwhm/3
        
        estimate = self.make_params()
        estimate['amplitude'].set(value=amplitude, min=0, max=100 * amplitude)
        estimate['fwhm'].set(value=fwhm, min=x_spacing/10, max=x_span)
        estimate['center'].set(value=center, min=min(x) - x_span / 2, max=max(x) + x_span / 2)
        estimate['offset'].set(value=offset, min=min(min(data.real),min(data.imag)) - data_span / 2, max=max(max(data.real),max(data.imag)) + data_span / 2 )
        estimate['amplitude_factor'].set(value=1, min=0, max=np.inf)
        estimate['IQphase'].set(value=IQphase,min=-np.pi, max=np.pi)
        return estimate
        
    @estimator('Peak')
    def estimate_peak(self, data, x):
        data, x = sort_check_data(data, x)
        offsetI,amplitudeI,centerI,fwhmI,[x_spacing,x_span,data_spanI] = single_peak_estimator(data.real, x, useMinMaxCenter=True)
        offsetQ,amplitudeQ,centerQ,fwhmQ,[x_spacing,x_span,data_spanQ] = single_peak_estimator(data.imag, x, useMinMaxCenter=True)
        data_span = max(data_spanI,data_spanQ)
        
        amplitude = np.sqrt(amplitudeI**2+amplitudeQ**2)
        IQphase = np.arctan2(amplitudeQ,amplitudeI)
        #
        center = centerI if amplitudeI>amplitudeQ else centerQ
        fwhm = fwhmI if amplitudeI>amplitudeQ else fwhmQ
        offset = (offsetI+offsetQ)/2
        
        # correct since it was obtained from the area assuming a single peak
        fwhm = fwhm/3
        
        estimate = self.make_params()
        estimate['amplitude'].set(value=amplitude, min=0, max=100 * amplitude)
        estimate['fwhm'].set(value=fwhm, min=x_spacing/10, max=x_span)
        estimate['center'].set(value=center, min=min(x) - x_span / 2, max=max(x) + x_span / 2)
        estimate['offset'].set(value=offset, min=min(min(data.real),min(data.imag)) - data_span / 2, max=max(max(data.real),max(data.imag)) + data_span / 2 )
        estimate['amplitude_factor'].set(value=1, min=0, max=np.inf)
        estimate['IQphase'].set(value=IQphase,min=-np.pi, max=np.pi)
        return estimate
        
    @estimator('Peak (no offset)')
    def estimate_peak_no_offset(self, data, x):
        estimate = self.estimate_peak(data, x)
        estimate['offset'].set(value=0, min=-np.inf, max=np.inf, vary=False)
        return estimate
        

class ODMR7PeakSpectrum_IQ(FitModelBase):
    """ 
    Seven peak Lorentzian for complex data.
    Using amplitude_factor for left and right peaks:
    amplitude: amplitude of central peak
    amplitude*amplitude_factor_1: amplitude of secondary peaks (ideally 2/3 of central peak)
    amplitude*amplitude_factor_2: amplitude of tertiary peaks (ideally 1/3 of central peak)
    amplitude*amplitude_factor_3: amplitude of extra peaks (harmonics, ideally equal to 0)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_param_hint('offset', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('center', value=1e-15, min=0, max=np.inf)
        self.set_param_hint('fwhm', value=0., min=0, max=np.inf)
        self.set_param_hint('hyperfine', value=2.16e6, min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude_factor_1', value=2/3, min=0, max=np.inf)
        self.set_param_hint('amplitude_factor_2', value=1/3, min=0, max=np.inf)
        self.set_param_hint('amplitude_factor_3', value=1/9, min=0, max=np.inf)
        self.set_param_hint('IQphase', value=0, min=-np.pi, max=np.pi)

    @staticmethod
    def _model_function(x, offset,amplitude,center,fwhm,hyperfine,
                        amplitude_factor_1,amplitude_factor_2,amplitude_factor_3,IQphase):
        xc0 = center - hyperfine*3
        xc1 = center - hyperfine*2
        xc2 = center - hyperfine
        xc3 = center
        xc4 = center + hyperfine
        xc5 = center + hyperfine*2
        xc6 = center + hyperfine*3
        #
        #a3 = amplitude
        #a2,a4 = amplitude*amplitude_factor_1,amplitude*amplitude_factor_1
        #a1,a5 = amplitude*amplitude_factor_2,amplitude*amplitude_factor_2
        #a0,a6 = amplitude*amplitude_factor_3,amplitude*amplitude_factor_3
        #
        #a0I,a1I,a2I,a3I,a4I = amplitude*amplitude_scaling*np.cos(IQphase)
        #a0Q,a1Q,a2Q,a3Q,a4Q = amplitude*amplitude_scaling*np.sin(IQphase)
        a3I = amplitude*np.cos(IQphase)
        a2I,a4I = a3I*amplitude_factor_1,a3I*amplitude_factor_1
        a1I,a5I = a3I*amplitude_factor_2,a3I*amplitude_factor_2
        a0I,a6I = a3I*amplitude_factor_3,a3I*amplitude_factor_3
        a3Q = amplitude*np.sin(IQphase)
        a2Q,a4Q = a3Q*amplitude_factor_1,a3Q*amplitude_factor_1
        a1Q,a5Q = a3Q*amplitude_factor_2,a3Q*amplitude_factor_2
        a0Q,a6Q = a3Q*amplitude_factor_3,a3Q*amplitude_factor_3
        
        real = func7Lorentz(x,offset,a0I,xc0,fwhm, a1I,xc1,fwhm, a2I,xc2,fwhm, a3I,xc3,fwhm, a4I,xc4,fwhm, a5I,xc5,fwhm, a6I,xc6,fwhm)
        imag = func7Lorentz(x,offset,a0Q,xc0,fwhm, a1Q,xc1,fwhm, a2Q,xc2,fwhm, a3Q,xc3,fwhm, a4Q,xc4,fwhm, a5Q,xc5,fwhm, a6Q,xc6,fwhm)
        return (real + 1j*imag).view()
    
    @estimator('Peak (center self correl)')
    def estimate_peak_selfcorr(self, data, x):
        data, x = sort_check_data(data, x)
        offsetI,amplitudeI,centerI,fwhmI,[x_spacing,x_span,data_spanI] = single_peak_estimator(data.real, x, useMinMaxCenter=False)
        offsetQ,amplitudeQ,centerQ,fwhmQ,[x_spacing,x_span,data_spanQ] = single_peak_estimator(data.imag, x, useMinMaxCenter=False)
        data_span = max(data_spanI,data_spanQ)
        
        amplitude = np.sqrt(amplitudeI**2+amplitudeQ**2)
        IQphase = np.arctan2(amplitudeQ,amplitudeI)
        #
        center = centerI if amplitudeI>amplitudeQ else centerQ
        fwhm = fwhmI if amplitudeI>amplitudeQ else fwhmQ
        offset = (offsetI+offsetQ)/2

        # correct since it was obtained from the area assuming a single peak
        fwhm = fwhm/3 # 1/3+2/3+1+2/3+1/3 = 3 (ignoring harmonics)

        estimate = self.make_params()
        estimate['amplitude'].set(value=amplitude, min=0, max=100 * amplitude)
        estimate['fwhm'].set(value=fwhm, min=x_spacing/10, max=x_span)
        estimate['center'].set(value=center, min=min(x) - x_span / 2, max=max(x) + x_span / 2)
        estimate['offset'].set(value=offset, min=min(min(data.real),min(data.imag)) - data_span / 2, max=max(max(data.real),max(data.imag)) + data_span / 2 )
        estimate['amplitude_factor_1'].set(value=2/3)
        estimate['amplitude_factor_2'].set(value=1/3)
        estimate['amplitude_factor_3'].set(value=1/9)
        estimate['IQphase'].set(value=IQphase,min=-np.pi, max=np.pi)
        return estimate
        
    @estimator('Peak')
    def estimate_peak(self, data, x):
        data, x = sort_check_data(data, x)
        offsetI,amplitudeI,centerI,fwhmI,[x_spacing,x_span,data_spanI] = single_peak_estimator(data.real, x)
        offsetQ,amplitudeQ,centerQ,fwhmQ,[x_spacing,x_span,data_spanQ] = single_peak_estimator(data.imag, x)
        data_span = max(data_spanI,data_spanQ)
        
        amplitude = np.sqrt(amplitudeI**2+amplitudeQ**2)
        IQphase = np.arctan2(amplitudeQ,amplitudeI)
        #
        center = centerI if amplitudeI>amplitudeQ else centerQ
        fwhm = fwhmI if amplitudeI>amplitudeQ else fwhmQ
        offset = (offsetI+offsetQ)/2

        # correct since it was obtained from the area assuming a single peak
        fwhm = fwhm/3 # 1/3+2/3+1+2/3+1/3 = 3 (ignoring harmonics)

        estimate = self.make_params()
        estimate['amplitude'].set(value=amplitude, min=0, max=100 * amplitude)
        estimate['fwhm'].set(value=fwhm, min=x_spacing/10, max=x_span)
        estimate['center'].set(value=center, min=min(x) - x_span / 2, max=max(x) + x_span / 2)
        estimate['offset'].set(value=offset, min=min(min(data.real),min(data.imag)) - data_span / 2, max=max(max(data.real),max(data.imag)) + data_span / 2 )
        estimate['amplitude_factor_1'].set(value=2/3)
        estimate['amplitude_factor_2'].set(value=1/3)
        estimate['amplitude_factor_3'].set(value=1/9)
        estimate['IQphase'].set(value=IQphase,min=-np.pi, max=np.pi)
        return estimate
        
    @estimator('Dip')
    def estimate_dip(self, data, x):
        estimate = self.estimate_peak(-data, x)
        estimate['offset'].set(value=-estimate['offset'].value,min=-estimate['offset'].max,max=-estimate['offset'].min)
        estimate['amplitude'].set(value=-estimate['amplitude'].value,min=-estimate['amplitude'].max,max=-estimate['amplitude'].min)
        return estimate

    @estimator('Peak (no offset)')
    def estimate_peak_no_offset(self, data, x):
        estimate = self.estimate_peak(data, x)
        estimate['offset'].set(value=0, min=-np.inf, max=np.inf, vary=False)
        return estimate

    @estimator('Dip (no offset)')
    def estimate_dip_no_offset(self, data, x):
        estimate = self.estimate_dip(data, x)
        estimate['offset'].set(value=0, min=-np.inf, max=np.inf, vary=False)
        return estimate


###################################################################################
## This is a copy of the model defined in  
## /home/nvuser/qudi_venv/lib/python3.10/site-packages/qudi/util/fit_models/sine.py
## Except from the estimator, which is built from ./analysisFunctions.py
###################################################################################
from lens.analysisFunctions import fftaux
class Sine(FitModelBase):
    """
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_param_hint('offset', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude', value=1., min=0., max=np.inf)
        self.set_param_hint('frequency', value=0., min=0., max=np.inf)
        self.set_param_hint('phase', value=0., min=-np.pi, max=np.pi)

    @staticmethod
    def _model_function(x, offset, amplitude, frequency, phase):
        return offset + amplitude * np.sin(2 * np.pi * frequency * x + phase)

    @estimator('default')
    def estimate(self, data, x):
        data, x = sort_check_data(data, x)
        x_span = abs(max(x) - min(x))
        offset = np.mean(data)

        estimate = self.estimate_no_offset(data - offset, x)
        if 1/(2 * estimate['frequency'].value) > x_span:
            estimate['offset'].set(value=offset, min=-np.inf, max=np.inf, vary=True)
        else:
            estimate['offset'].set(value=offset, min=min(data), max=max(data), vary=True)
        return estimate

    @estimator('No Offset')
    def estimate_no_offset(self, data, x):
        data, x = sort_check_data(data, x)
        data_span = abs(max(data) - min(data))

        _, _, p0 = fftaux(x,data, sP=False, return_estim=True) # estimators
        
        offset,amplitude,frequency,phase = p0
        phase = -phase

        estimate = self.make_params()
        estimate['frequency'].set(value=frequency, min=0, max=np.inf, vary=True)
        estimate['amplitude'].set(value=amplitude, min=0, max=10 * data_span, vary=True)
        estimate['phase'].set(value=phase, min=-np.pi, max=np.pi, vary=True)
        estimate['offset'].set(value=0, min=-np.inf, max=np.inf, vary=False)
        return estimate



