# This is for functions that you use repeatedly during the analysis of the data
# that are compatible with the qudi format for storing data and for fitting data 

import numpy as np
import matplotlib.pyplot as plt

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
amplitude_scaling = np.array([1,2,3,2,1])
from lens.analysisFunctions import funcLorentz
def func5Lorentz(x,y0,a,xc,w, a1,xc1,w1, a2,xc2,w2, a3,xc3,w3, a4,xc4,w4):
    return y0 + funcLorentz(x,0,a,xc,w) + funcLorentz(x,0,a1,xc1,w1) + funcLorentz(x,0,a2,xc2,w2) + funcLorentz(x,0,a3,xc3,w3) + funcLorentz(x,0,a4,xc4,w4)
def func5peakspectrum(x, y0,a,xc,w):
    xc0 = xc - hyperfine*2
    xc1 = xc - hyperfine
    xc2 = xc
    xc3 = xc + hyperfine
    xc4 = xc + hyperfine*2
    a0,a1,a2,a3,a4 = a*amplitude_scaling
    return func5Lorentz(x,y0,a0,xc0,w, a1,xc1,w, a2,xc2,w, a3,xc3,w, a4,xc4,w)
#
def funcDerivativeLorentz(x,y0,a,xc,w):
    return y0 - a*(2/np.pi)*w*8*(x-xc)/(4*(x-xc)**2+w**2)**2
def func5DerivativeLorentz(x,y0,a,xc,w, a1,xc1,w1, a2,xc2,w2, a3,xc3,w3, a4,xc4,w4):
    return y0 + funcDerivativeLorentz(x,0,a,xc,w) + funcDerivativeLorentz(x,0,a1,xc1,w1) + funcDerivativeLorentz(x,0,a2,xc2,w2) + funcDerivativeLorentz(x,0,a3,xc3,w3) + funcDerivativeLorentz(x,0,a4,xc4,w4)
def func5peakderivativespectrum(x, y0,a,xc,w):
    xc0 = xc - hyperfine*2
    xc1 = xc - hyperfine
    xc2 = xc
    xc3 = xc + hyperfine
    xc4 = xc + hyperfine*2
    a0,a1,a2,a3,a4 = a*amplitude_scaling
    return func5DerivativeLorentz(x,y0,a0,xc0,w, a1,xc1,w, a2,xc2,w, a3,xc3,w, a4,xc4,w)




##
## To import the qudi helpers file: /home/santiago/qudi_venv/lib/python3.10/site-packages/qudi/util/fit_models/helpers.py
## we have created a symbolic link of qudi inside lens module
##
from lens.qudi_link.util.fit_models.helpers import *

    
##
## From /home/santiago/qudi_venv/lib/python3.10/site-packages/qudi/util/fit_models/model.py
##
import inspect
from lmfit import Model, CompositeModel
from abc import ABCMeta, abstractmethod

def estimator(name):
    assert isinstance(name, str) and name, 'estimator name must be non-empty str'

    def _decorator(func):
        assert callable(func), 'estimator must be callable'
        params = tuple(inspect.signature(func).parameters)
        assert len(params) == 3, \
            'estimator must be bound method with 2 positional parameters. First parameter is the ' \
            'y data array to use and second parameter is the corresponding independent variable.'
        func._estimator_name = name
        func._estimator_independent_var = params[2]
        return func

    return _decorator

class FitModelMeta(ABCMeta):
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)

        # collect marked estimator methods in a dict "_estimators" and attach it to created class.
        # NOTE: Estimators defined in base classes will not be taken into account. If you want to
        # do inheritance shenanigans with these fit model classes, you need to manually handle this
        # in the implementation.
        # Generally one can not assume parent estimators to be valid for a subclass.
        cls._estimators = {attr._estimator_name: attr for attr in attrs.values() if
                           hasattr(attr, '_estimator_name')}
        independent_vars = {e._estimator_independent_var for e in cls._estimators.values()}
        assert len(independent_vars) < 2, \
            'More than one independent variable name encountered in estimators. Use only the ' \
            'independent variable name that has been used in the Models "_model_function".'


class FitModelBase(Model, metaclass=FitModelMeta):
    """ ToDo: Document
    """

    def __init__(self, **kwargs):
        kwargs['name'] = self.__class__.__name__
        super().__init__(self._model_function, **kwargs)
        assert len(self.independent_vars) == 1, \
            'Qudi fit models must contain exactly 1 independent variable.'
        # Shadow FitModelBase._estimators with a similar dict containing the bound method objects.
        # This instance-level dict has read-only access via property "estimators"
        self._estimators = {name: getattr(self, e.__name__) for name, e in self._estimators.items()}

    @property
    def estimators(self):
        """ Read-only dict property holding available estimator names as keys and the corresponding
        estimator methods as values.

        @return dict: Available estimator methods (values) with corresponding names (keys)
        """
        return self._estimators.copy()

    @staticmethod
    @abstractmethod
    def _model_function(x, **kwargs):
        """ ToDo: Document
        """
        raise NotImplementedError('FitModel object must implement staticmethod "_model_function".')
        
##
## From /home/santiago/qudi_venv/lib/python3.10/site-packages/qudi/util/fit_models/lorentzian.py
##
def multiple_lorentzian(x, centers, sigmas, amplitudes):
    """ Mathematical definition of the sum of multiple (physical) Lorentzian functions without any
    bias.

    WARNING: iterable parameters "centers", "sigmas" and "amplitudes" must have same length.

    @param float x: The independent variable to calculate lorentz(x)
    @param iterable centers: Iterable containing center positions for all lorentzians
    @param iterable sigmas: Iterable containing sigmas for all lorentzians
    @param iterable amplitudes: Iterable containing amplitudes for all lorentzians
    """
    assert len(centers) == len(sigmas) == len(amplitudes)
    return sum(amp * sig ** 2 / ((x - c) ** 2 + sig ** 2) for c, sig, amp in
               zip(centers, sigmas, amplitudes))

class Lorentzian(FitModelBase):
    """
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_param_hint('offset', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude', value=0., min=0., max=np.inf)
        self.set_param_hint('center', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('sigma', value=0., min=0., max=np.inf)

    @staticmethod
    def _model_function(x, offset, center, sigma, amplitude):
        return offset + multiple_lorentzian(x, (center,), (sigma,), (amplitude,))

    @estimator('Peak')
    def estimate_peak(self, data, x):
        data, x = sort_check_data(data, x)
        # Smooth data
        filter_width = max(1, int(round(len(x) / 20)))
        data_smoothed, _ = smooth_data(data, filter_width)
        data_smoothed, offset = correct_offset_histogram(data_smoothed, bin_width=2 * filter_width)

        # determine peak position
        center = x[np.argmax(data_smoothed)]

        # calculate amplitude
        amplitude = abs(max(data_smoothed))

        # according to the derived formula, calculate sigma. The crucial part is here that the
        # offset was estimated correctly, then the area under the curve is calculated correctly:
        numerical_integral = np.trapz(data_smoothed, x)
        sigma = abs(numerical_integral / (np.pi * amplitude))

        x_spacing = min(abs(np.ediff1d(x)))
        x_span = abs(x[-1] - x[0])
        data_span = abs(max(data) - min(data))

        estimate = self.make_params()
        estimate['amplitude'].set(value=amplitude, min=0, max=2 * amplitude)
        estimate['sigma'].set(value=sigma, min=x_spacing, max=x_span)
        estimate['center'].set(value=center, min=min(x) - x_span / 2, max=max(x) + x_span / 2)
        estimate['offset'].set(
            value=offset, min=min(data) - data_span / 2, max=max(data) + data_span / 2
        )
        return estimate

    @estimator('Dip')
    def estimate_dip(self, data, x):
        estimate = self.estimate_peak(-data, x)
        estimate['offset'].set(value=-estimate['offset'].value,
                               min=-estimate['offset'].max,
                               max=-estimate['offset'].min)
        estimate['amplitude'].set(value=-estimate['amplitude'].value,
                                  min=-estimate['amplitude'].max,
                                  max=-estimate['amplitude'].min)
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


class func5peakspectrum_Lorentzian(FitModelBase):
    """
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_param_hint('offset', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude', value=0., min=0., max=np.inf)
        self.set_param_hint('center', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('sigma', value=0., min=0., max=np.inf)

    @staticmethod
    def _model_function(x, offset, center, sigma, amplitude):
        return func5peakspectrum(x,offset,amplitude*sigma*np.pi/2,center,sigma)

    @estimator('Peak')
    def estimate_peak(self, data, x):
        data, x = sort_check_data(data, x)
        # Smooth data
        filter_width = max(1, int(round(len(x) / 20)))
        data_smoothed, _ = smooth_data(data, filter_width)
        data_smoothed, offset = correct_offset_histogram(data_smoothed, bin_width=2 * filter_width)

        # determine peak position
        center = x[np.argmax(data_smoothed)]

        # calculate amplitude
        amplitude = abs(max(data_smoothed))

        # according to the derived formula, calculate sigma. The crucial part is here that the
        # offset was estimated correctly, then the area under the curve is calculated correctly:
        numerical_integral = np.trapz(data_smoothed, x)
        sigma = abs(numerical_integral / (np.pi * amplitude))

        x_spacing = min(abs(np.ediff1d(x)))
        x_span = abs(x[-1] - x[0])
        data_span = abs(max(data) - min(data))

        estimate = self.make_params()
        estimate['amplitude'].set(value=amplitude, min=0, max=2 * amplitude)
        estimate['sigma'].set(value=sigma, min=x_spacing, max=x_span)
        estimate['center'].set(value=center, min=min(x) - x_span / 2, max=max(x) + x_span / 2)
        estimate['offset'].set(
            value=offset, min=min(data) - data_span / 2, max=max(data) + data_span / 2
        )
        return estimate

    @estimator('Dip')
    def estimate_dip(self, data, x):
        estimate = self.estimate_peak(-data, x)
        estimate['offset'].set(value=-estimate['offset'].value,
                               min=-estimate['offset'].max,
                               max=-estimate['offset'].min)
        estimate['amplitude'].set(value=-estimate['amplitude'].value,
                                  min=-estimate['amplitude'].max,
                                  max=-estimate['amplitude'].min)
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

########################
########################
########################

class DoubleLorentzian(FitModelBase):
    """ ToDo: Document
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_param_hint('offset', value=0, min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude_1', value=0, min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude_2', value=0, min=-np.inf, max=np.inf)
        self.set_param_hint('center_1', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('center_2', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('sigma_1', value=0., min=0., max=np.inf)
        self.set_param_hint('sigma_2', value=0., min=0., max=np.inf)

    @staticmethod
    def _model_function(x, offset, center_1, center_2, sigma_1, sigma_2, amplitude_1, amplitude_2):
        return offset + multiple_lorentzian(x,
                                                (center_1, center_2),
                                                (sigma_1, sigma_2),
                                                (amplitude_1, amplitude_2))

    @estimator('Peaks')
    def estimate_peaks(self, data, x):
        data, x = sort_check_data(data, x)
        data_smoothed, filter_width = smooth_data(data)
        leveled_data_smooth, offset = correct_offset_histogram(data_smoothed,
                                                               bin_width=2 * filter_width)
        estimate, limits = estimate_double_peaks(leveled_data_smooth, x, filter_width)

        params = self.make_params()
        params['amplitude_1'].set(value=estimate['height'][0],
                                  min=limits['height'][0][0],
                                  max=limits['height'][0][1])
        params['amplitude_2'].set(value=estimate['height'][1],
                                  min=limits['height'][1][0],
                                  max=limits['height'][1][1])
        params['center_1'].set(value=estimate['center'][0],
                               min=limits['center'][0][0],
                               max=limits['center'][0][1])
        params['center_2'].set(value=estimate['center'][1],
                               min=limits['center'][1][0],
                               max=limits['center'][1][1])
        params['sigma_1'].set(value=estimate['fwhm'][0] / 2.3548,
                              min=limits['fwhm'][0][0] / 2.3548,
                              max=limits['fwhm'][0][1] / 2.3548)
        params['sigma_2'].set(value=estimate['fwhm'][1] / 2.3548,
                              min=limits['fwhm'][1][0] / 2.3548,
                              max=limits['fwhm'][1][1] / 2.3548)
        return params

    @estimator('Dips')
    def estimate_dips(self, data, x):
        estimate = self.estimate_peaks(-data, x)
        estimate['offset'].set(value=-estimate['offset'].value,
                               min=-estimate['offset'].max,
                               max=-estimate['offset'].min)
        estimate['amplitude_1'].set(value=-estimate['amplitude_1'].value,
                                    min=-estimate['amplitude_1'].max,
                                    max=-estimate['amplitude_1'].min)
        estimate['amplitude_2'].set(value=-estimate['amplitude_2'].value,
                                    min=-estimate['amplitude_2'].max,
                                    max=-estimate['amplitude_2'].min)
        return estimate
    
class TripleLorentzian(FitModelBase):
    """ ToDo: Document
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_param_hint('offset', value=0, min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude_1', value=0, min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude_2', value=0, min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude_3', value=0, min=-np.inf, max=np.inf)
        self.set_param_hint('center_1', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('center_2', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('center_3', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('sigma_1', value=0., min=0., max=np.inf)
        self.set_param_hint('sigma_2', value=0., min=0., max=np.inf)
        self.set_param_hint('sigma_3', value=0., min=0., max=np.inf)

    @staticmethod
    def _model_function(x, offset, center_1, center_2, center_3, sigma_1, sigma_2, sigma_3,
                        amplitude_1, amplitude_2, amplitude_3):
        return offset + multiple_lorentzian(x,
                                            (center_1, center_2, center_3),
                                            (sigma_1, sigma_2, sigma_3),
                                            (amplitude_1, amplitude_2, amplitude_3))

    @estimator('Peaks')
    def estimate_peaks(self, data, x):
        data, x = sort_check_data(data, x)
        data_smoothed, filter_width = smooth_data(data)
        leveled_data_smooth, offset = correct_offset_histogram(data_smoothed,
                                                               bin_width=2 * filter_width)
        estimate, limits = estimate_triple_peaks(leveled_data_smooth, x, filter_width)

        params = self.make_params()
        params['amplitude_1'].set(value=estimate['height'][0],
                                  min=limits['height'][0][0],
                                  max=limits['height'][0][1])
        params['amplitude_2'].set(value=estimate['height'][1],
                                  min=limits['height'][1][0],
                                  max=limits['height'][1][1])
        params['amplitude_3'].set(value=estimate['height'][2],
                                  min=limits['height'][2][0],
                                  max=limits['height'][2][1])
        params['center_1'].set(value=estimate['center'][0],
                               min=limits['center'][0][0],
                               max=limits['center'][0][1])
        params['center_2'].set(value=estimate['center'][1],
                               min=limits['center'][1][0],
                               max=limits['center'][1][1])
        params['center_3'].set(value=estimate['center'][2],
                               min=limits['center'][2][0],
                               max=limits['center'][2][1])
        params['sigma_1'].set(value=estimate['fwhm'][0] / 2.3548,
                              min=limits['fwhm'][0][0] / 2.3548,
                              max=limits['fwhm'][0][1] / 2.3548)
        params['sigma_2'].set(value=estimate['fwhm'][1] / 2.3548,
                              min=limits['fwhm'][1][0] / 2.3548,
                              max=limits['fwhm'][1][1] / 2.3548)
        params['sigma_3'].set(value=estimate['fwhm'][2] / 2.3548,
                              min=limits['fwhm'][2][0] / 2.3548,
                              max=limits['fwhm'][2][1] / 2.3548)
        return params

    @estimator('Dips')
    def estimate_dips(self, data, x):
        estimate = self.estimate_peaks(-data, x)
        estimate['offset'].set(value=-estimate['offset'].value,
                               min=-estimate['offset'].max,
                               max=-estimate['offset'].min)
        estimate['amplitude_1'].set(value=-estimate['amplitude_1'].value,
                                    min=-estimate['amplitude_1'].max,
                                    max=-estimate['amplitude_1'].min)
        estimate['amplitude_2'].set(value=-estimate['amplitude_2'].value,
                                    min=-estimate['amplitude_2'].max,
                                    max=-estimate['amplitude_2'].min)
        estimate['amplitude_3'].set(value=-estimate['amplitude_3'].value,
                                    min=-estimate['amplitude_3'].max,
                                    max=-estimate['amplitude_3'].min)
        return estimate
    
    
class FiveLorentzian(FitModelBase):
    """ ToDo: Document
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_param_hint('offset', value=0, min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude_1', value=0, min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude_2', value=0, min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude_3', value=0, min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude_4', value=0, min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude_5', value=0, min=-np.inf, max=np.inf)
        self.set_param_hint('center_1', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('center_2', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('center_3', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('center_4', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('center_5', value=0., min=-np.inf, max=np.inf)
        self.set_param_hint('sigma_1', value=0., min=0., max=np.inf)
        self.set_param_hint('sigma_2', value=0., min=0., max=np.inf)
        self.set_param_hint('sigma_3', value=0., min=0., max=np.inf)
        self.set_param_hint('sigma_4', value=0., min=0., max=np.inf)
        self.set_param_hint('sigma_5', value=0., min=0., max=np.inf)

    @staticmethod
    def _model_function(x, offset, center_1, center_2, center_3, center_4, center_5, 
                        sigma_1, sigma_2, sigma_3, sigma_4, sigma_5,
                        amplitude_1, amplitude_2, amplitude_3, amplitude_4, amplitude_5):
        return offset + multiple_lorentzian(x,
                                            (center_1, center_2, center_3, center_4, center_5),
                                            (sigma_1, sigma_2, sigma_3, sigma_4, sigma_5),
                                            (amplitude_1, amplitude_2, amplitude_3, amplitude_4, amplitude_5))

    @estimator('Peaks')
    def estimate_peaks(self, data, x):
        data, x = sort_check_data(data, x)
        data_smoothed, filter_width = smooth_data(data)
        leveled_data_smooth, offset = correct_offset_histogram(data_smoothed,
                                                               bin_width=2 * filter_width)
        estimate, limits = estimate_five_peaks(leveled_data_smooth, x, filter_width)

        params = self.make_params()
        params['amplitude_1'].set(value=estimate['height'][0],
                                  min=limits['height'][0][0],
                                  max=limits['height'][0][1])
        params['amplitude_2'].set(value=estimate['height'][1],
                                  min=limits['height'][1][0],
                                  max=limits['height'][1][1])
        params['amplitude_3'].set(value=estimate['height'][2],
                                  min=limits['height'][2][0],
                                  max=limits['height'][2][1])
        params['amplitude_4'].set(value=estimate['height'][3],
                                  min=limits['height'][3][0],
                                  max=limits['height'][3][1])
        params['amplitude_5'].set(value=estimate['height'][4],
                                  min=limits['height'][4][0],
                                  max=limits['height'][4][1])
        params['center_1'].set(value=estimate['center'][0],
                               min=limits['center'][0][0],
                               max=limits['center'][0][1])
        params['center_2'].set(value=estimate['center'][1],
                               min=limits['center'][1][0],
                               max=limits['center'][1][1])
        params['center_3'].set(value=estimate['center'][2],
                               min=limits['center'][2][0],
                               max=limits['center'][2][1])
        params['center_4'].set(value=estimate['center'][3],
                               min=limits['center'][3][0],
                               max=limits['center'][3][1])
        params['center_5'].set(value=estimate['center'][4],
                               min=limits['center'][4][0],
                               max=limits['center'][4][1])
        params['sigma_1'].set(value=estimate['fwhm'][0] / 2.3548,
                              min=limits['fwhm'][0][0] / 2.3548,
                              max=limits['fwhm'][0][1] / 2.3548)
        params['sigma_2'].set(value=estimate['fwhm'][1] / 2.3548,
                              min=limits['fwhm'][1][0] / 2.3548,
                              max=limits['fwhm'][1][1] / 2.3548)
        params['sigma_3'].set(value=estimate['fwhm'][2] / 2.3548,
                              min=limits['fwhm'][2][0] / 2.3548,
                              max=limits['fwhm'][2][1] / 2.3548)
        params['sigma_4'].set(value=estimate['fwhm'][3] / 2.3548,
                              min=limits['fwhm'][3][0] / 2.3548,
                              max=limits['fwhm'][3][1] / 2.3548)
        params['sigma_5'].set(value=estimate['fwhm'][4] / 2.3548,
                              min=limits['fwhm'][4][0] / 2.3548,
                              max=limits['fwhm'][4][1] / 2.3548)
        return params

    @estimator('Dips')
    def estimate_dips(self, data, x):
        estimate = self.estimate_peaks(-data, x)
        estimate['offset'].set(value=-estimate['offset'].value,
                               min=-estimate['offset'].max,
                               max=-estimate['offset'].min)
        estimate['amplitude_1'].set(value=-estimate['amplitude_1'].value,
                                    min=-estimate['amplitude_1'].max,
                                    max=-estimate['amplitude_1'].min)
        estimate['amplitude_2'].set(value=-estimate['amplitude_2'].value,
                                    min=-estimate['amplitude_2'].max,
                                    max=-estimate['amplitude_2'].min)
        estimate['amplitude_3'].set(value=-estimate['amplitude_3'].value,
                                    min=-estimate['amplitude_3'].max,
                                    max=-estimate['amplitude_3'].min)
        estimate['amplitude_4'].set(value=-estimate['amplitude_4'].value,
                                    min=-estimate['amplitude_4'].max,
                                    max=-estimate['amplitude_4'].min)
        estimate['amplitude_5'].set(value=-estimate['amplitude_5'].value,
                                    min=-estimate['amplitude_5'].max,
                                    max=-estimate['amplitude_5'].min)
        return estimate
    
