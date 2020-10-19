# This is for functions to re-normalize the data to spin-lock maximum and minimum levels
# content with description                                                      -(line):
# Parameters for ref_max e ref_min definition                                      18
# Functions to re-normalize the data                                               53
# Main function returning the new re-normalization with errors                     65



import numpy as np
import matplotlib.pyplot as plt

from lens.analysisFunctions import weightAvg
import lens.fit_testing.decay_fit_methods as dfm


# Reference levels from spin-lock
#left_to_left_mean = 0.9443 # Parameters from 2019-12-17 measurements with 3 read outs @ 0.5 mW
#left_to_leftE_mean = 0.0641
#up_to_up_mean = 0.9149
#up_to_upE_mean = 0.0362
#down_to_down_mean = 0.9774
#down_to_downE_mean = 0.0484
#left_to_up_mean = 0.2827
#left_to_upE_mean = 0.0576
#left_to_down_mean = 0.2517
#left_to_downE_mean = 0.0328
#up_to_left_mean = 0.2035
#up_to_leftE_mean = 0.0359
#up_to_down_mean = 0.1973
#up_to_downE_mean = 0.0302
#down_to_up_mean = 0.2083
#down_to_upE_mean = 0.0385
#down_to_left_mean = 0.2143
#down_to_leftE_mean = 0.0304
offsetDefault = np.array([1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
offset05mW = np.array([0.9443,0.0641,0.9149,0.0362,0.9774,0.0484,0.2827,0.0576,0.2517,
                       0.0328,0.2035,0.0359,0.1973,0.0302,0.2083,0.0385,0.2143,0.0304])
offset03mW = np.array([0.8871,0.0446,0.9160,0.0371,0.9005,0.0396,0.1067,0.0647,0.1320,
                       0.0570,0.1325,0.0639,0.0655,0.0637,0.1236,0.0620,0.0680,0.0758]) #original
#offset03mW = np.array([0.8871,0.02,91.60,0.02,0.9005,0.02,0.1067,0.02,0.1320,
#                       0.02,0.1325,0.02,0.0655,0.02,0.1236,0.02,0.0680,0.02])

#def weightAvg(x,s):
#    mu = (x/s**2).sum()/(1/s**2).sum()
#    # The error it will return is sum in quadrature of  σ_w & σ_{\bar{x}}
#    # σ_w  (weighted sample variance):
#    wsv = np.sqrt( ((x**2/s**2).sum()/(1/s**2).sum()-mu**2)/(len(x)-1) )
#    # σ_{\bar{x}} (standard error of the weighted mean (with variance weights))
#    sewm = np.sqrt(1/(1/s**2).sum())
#    return mu, np.sqrt(wsv**2+sewm**2)

def refNorm(power):
    if power == '0.5':
        (left_to_left_mean,left_to_leftE_mean,up_to_up_mean,up_to_upE_mean,down_to_down_mean,down_to_downE_mean,
         left_to_up_mean,left_to_upE_mean,left_to_down_mean,left_to_downE_mean,
         up_to_left_mean,up_to_leftE_mean,up_to_down_mean,up_to_downE_mean,
         down_to_up_mean,down_to_upE_mean,down_to_left_mean,down_to_leftE_mean) = offset05mW
    elif power == '0.3':
        (left_to_left_mean,left_to_leftE_mean,up_to_up_mean,up_to_upE_mean,down_to_down_mean,down_to_downE_mean,
         left_to_up_mean,left_to_upE_mean,left_to_down_mean,left_to_downE_mean,
         up_to_left_mean,up_to_leftE_mean,up_to_down_mean,up_to_downE_mean,
         down_to_up_mean,down_to_upE_mean,down_to_left_mean,down_to_leftE_mean) = offset03mW
    else:
        print('WARNING: no re-normalization data for power = ',power)
        (left_to_left_mean,left_to_leftE_mean,up_to_up_mean,up_to_upE_mean,down_to_down_mean,down_to_downE_mean,
         left_to_up_mean,left_to_upE_mean,left_to_down_mean,left_to_downE_mean,
         up_to_left_mean,up_to_leftE_mean,up_to_down_mean,up_to_downE_mean,
         down_to_up_mean,down_to_upE_mean,down_to_left_mean,down_to_leftE_mean) = offsetDefault
    ref_max, ref_maxE = weightAvg(np.array([left_to_left_mean, up_to_up_mean, down_to_down_mean]),
                                  np.array([left_to_leftE_mean, up_to_upE_mean, down_to_downE_mean]))
    ref_min, ref_minE = weightAvg(np.array([left_to_up_mean, left_to_down_mean, up_to_left_mean, up_to_down_mean,
                                            down_to_up_mean, down_to_left_mean]),
                                  np.array([left_to_upE_mean, left_to_downE_mean, up_to_leftE_mean, up_to_downE_mean,
                                            down_to_upE_mean, down_to_leftE_mean]))
    return ref_max, ref_maxE, ref_min, ref_minE

# calculates the normalized population when |up> or |down> was prepared
def norm_pop(pops_raw, pops_rawE, power):
    ref_max,ref_maxE,ref_min, ref_minE = refNorm(power)
    pops_norm = (pops_raw - ref_min)/(ref_max - ref_min)
    #pops_normE = (pops_rawE + ref_minE)/(ref_max - ref_min)+(pops_raw + ref_min)*(ref_maxE - ref_minE)/(ref_max - ref_min)**2
    pops_normE = np.sqrt(pops_rawE**2 + ref_minE**2)/(ref_max - ref_min)+(pops_raw - ref_min)*np.sqrt(ref_maxE**2 + ref_minE**2)/(ref_max - ref_min)**2
    return pops_norm, pops_normE

# calculates the normalized population assuming an offset shift of minimum level
def norm_pop_sym(pops_raw, pops_rawE, ref_min, ref_minE, power):
    ref_max,ref_maxE,*roba = refNorm(power)
    pops_norm = (pops_raw - ref_min/2)/(ref_max - ref_min/2)
    #pops_normE = (pops_rawE + ref_minE)/(ref_max - ref_min/2)+(pops_raw + ref_min/2)*(ref_maxE - ref_minE)/(ref_max - ref_min/2)**2
    pops_normE = np.sqrt(pops_rawE**2 + ref_minE**2/4)/(ref_max - ref_min/2)+(pops_raw - ref_min/2)*np.sqrt(ref_maxE**2 + ref_minE**2/4)/(ref_max - ref_min/2)**2
    return pops_norm, pops_normE


def reNormalize(data, normType='spinLock', power='0.5'):
    """
    data[0] = normalized data
    data[1] = errors of normalized data
    data[2] = time vector
    normType = 'spinLock' or 'popSum' defines if consider all spin-lock references or take the ref_min from populations sum
    power = defines the max and minimum offset for re-normalization
    """
    if np.shape(data[0])[1]/3%1 == 0:
        dataset = int(np.shape(data[0])[1]/3)
        to_left = data[0][0][0:dataset]
        to_leftE = data[1][0][0:dataset]
        to_up = data[0][0][dataset:2*dataset]
        to_upE = data[1][0][dataset:2*dataset]
        to_down = data[0][0][2*dataset:]
        to_downE = data[1][0][2*dataset:]
    else:
        dataset = int((np.shape(data[0])[1]-2)/3)
        to_left = data[0][0][0:dataset]
        to_leftE = data[1][0][0:dataset]
        to_up = data[0][0][dataset+1:2*dataset+1]
        to_upE = data[1][0][dataset+1:2*dataset+1]
        to_down = data[0][0][2*dataset+1:-1]
        to_downE = data[1][0][2*dataset+1:-1]
    xdata = data[2][0:dataset]
    pop_raw_summed = to_up + to_left + to_down
    pop_rawE_summed = to_upE + to_leftE + to_downE
    exp_fit_raw_summed = dfm.make_decayexponential_fit(xdata, pop_raw_summed, dfm.estimate_decayexponential)
    pop_raw_summed_fit = exp_fit_raw_summed.best_fit

    # normalize population
    pops_raw = np.array([to_up, to_left, to_down])
    pops_rawE = np.array([to_upE, to_leftE, to_downE])
    if normType == 'spinLock':
        pops_norm,pops_normE = norm_pop(pops_raw, pops_rawE, power)
    elif normType == 'popSum':
        pops_norm = np.zeros(np.shape(pops_raw))
        pops_normE = np.zeros(np.shape(pops_rawE))
        ref_max,ref_maxE,*roba = refNorm(power)
        for ii,pop in enumerate(pop_raw_summed_fit):
            pops_norm[:,ii],pops_normE[:,ii] = norm_pop_sym(pops_raw[:,ii], pops_rawE[:,ii], pop-ref_max, ref_maxE, power)
    pop_norm_summed = pops_norm[0] + pops_norm[1] + pops_norm[2]
    pop_normE_summed = np.sqrt((pops_normE[0]**2 + pops_normE[1]**2 + pops_normE[2]**2)/3)

    # exponential fit re-normalized data:
    exp_fit_summed = dfm.make_decayexponential_fit(xdata, pop_norm_summed, dfm.estimate_decayexponential)
    pop_summed_fit = exp_fit_summed.best_fit

    print('offset of fit of summed normalized populations', exp_fit_raw_summed.best_values['offset'])

    plt.figure(figsize=[15,5])
    plt.subplot(1,2,1)
    plt.errorbar(xdata, pop_raw_summed,yerr=pop_rawE_summed, fmt='o-', label='sum raw')
    plt.errorbar(xdata, pop_norm_summed,yerr=pop_normE_summed, fmt='o-', label='sum normalized')
    plt.plot(xdata, pop_raw_summed_fit, 'b', label='sum raw fit')
    plt.plot(xdata, pop_summed_fit, 'm', label='sum normalized fit')
    plt.xlabel('time (µs)')
    plt.ylabel('summed normalized population')
    plt.legend()
    plt.grid()

    plt.subplot(1,2,2)
    plt.errorbar(xdata, pops_norm[0],yerr=pops_normE[0], fmt='bo-', label=r'$\left|\uparrow\right>$')
    plt.errorbar(xdata, pops_norm[1],yerr=pops_normE[1], fmt='ro-', label=r'$\left|\leftarrow\right>$')
    plt.errorbar(xdata, pops_norm[2],yerr=pops_normE[2], fmt='go-', label=r'$\left|\downarrow\right>$')
    plt.xlabel('time (µs)')
    plt.ylabel('normalized population')
    plt.legend()
    plt.grid()
    plt.show()

    return xdata,pops_norm,pops_normE,pop_raw_summed_fit
