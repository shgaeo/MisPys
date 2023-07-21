
import numpy as np
import matplotlib.pyplot as plt

from lens.openDat import openDatFile
from lens.analysisFunctions import *

#globalskRows = 6 #LENS
globalskRows = 0 #QEG

# to plot the data and their moving average
def signals_movAvgWeight(data,windowSize,sp=False,titleText='',xText='',xfact=1e6,yfact=1,keep_edges=True,labels=None,n_signals=1,n_refs=0):
    """
    n_signals = 1,2,3,... number of signals
    n_refs    = 0,1,2 number of references
    """
    if not(n_refs in [0,1,2]):
        print('Error: n_refs must be 0 (no references), 1 (one reference) or 2 (two references)')
        return -1
    mask0 = np.arange(-(n_signals+n_refs),0)*2
    xdata=data[:,0]*xfact
    dat1=data[:,mask0]*yfact
    err1=data[:,(mask0+1)]*yfact
    
    if windowSize==-1:
        ma_all = 'no moving average'
    elif windowSize==0:
        ma_all = 'no moving average'
    else:
        ma_all = np.zeros([n_signals+n_refs,2,len(xdata)])
        for i in range(n_signals+n_refs):
            ma_all[i] = movAvgWeight(dat1[:,i],err1[:,i],windowSize)
            if keep_edges:
                # the moving average returns nan values in the firts and last positions, we substitute 
                # these nan values for original non averaged values
                mask = np.isnan(ma_all[i,0])
                ma_all[i,0,mask] = (dat1[:,i])[mask]
                ma_all[i,1,mask] = (err1[:,i])[mask]
    if sp:
        f,ax = plt.subplots(figsize=[6.4*2/3,4.8*2/3])
        for i in range(n_signals+n_refs):
            ax.errorbar(xdata,dat1[:,i],yerr=err1[:,i],fmt='.',ls=':')
            if windowSize!=0:
                ax.plot(xdata,ma_all[i][0],'-',lw=2,color='C'+str(i))
        #
        ax.set_xlabel(xText,fontsize=12)
        ax.set_ylabel('$\mathrm{kcps}$',fontsize=12)
        ax.set_title(titleText,fontsize=12)
        if not(labels is None):
            ax.legend(labels,loc=0)
        plt.tight_layout()
        plt.show()
    #
    return [xdata,dat1,err1,ma_all]

# to normalize the data
def signal_normalized(file_path,skRows=globalskRows,mov_avg_windowSize=4,n_signals=1,n_refs=0,sp=False,xfact=1e6,titleText='',xText='',labels=None):
    '''
    default function to fit a gaussian to the pulsed_esr sequence
    mov_avg_windowSize= even positive number: number of points for moving average ; 0 : raw data normalized ; -1 : normalize with mean value of references
    '''
    # 
    #print('\n',file_path[1+file_path.rfind('/'):])
    data=openDatFile(file_path,skRows=skRows)
    xdat,dat,err,movavg = signals_movAvgWeight(data,mov_avg_windowSize,sp=sp,titleText=titleText,xText=xText,xfact=xfact,keep_edges=True,labels=None,n_signals=n_signals,n_refs=n_refs)
    #
    ndat   = np.zeros([n_signals,len(xdat)])
    errdat = np.zeros([n_signals,len(xdat)])
    if n_refs==0:
        ndat,errdat = dat.transpose(),err.transpose()
    elif n_refs==1:
        if mov_avg_windowSize==-1:
            ref_0 = dat[:,0].mean()
        elif mov_avg_windowSize==0:
            ref_0 = dat[:,0]
        else:
            ref_0 = movavg[0,0]
        # Normalize signals to ms=0 reference
        for i in range(n_signals):
            ndat[i]   = (dat[:,n_refs+i])/(ref_0) - 1
            errdat[i] = (err[:,n_refs+i])/(ref_0)
    else:
        if mov_avg_windowSize==-1:
            ref_0 = dat[:,0].mean()
            ref_1 = dat[:,1].mean()
        elif mov_avg_windowSize==0:
            ref_0 = dat[:,0]
            ref_1 = dat[:,1]
        else:
            ref_0 = movavg[0,0]
            ref_1 = movavg[1,0]
        # Normalize signals to ms=0 reference
        for i in range(n_signals):
            ndat[i]   = (dat[:,n_refs+i]-ref_1)/(ref_0-ref_1)
            errdat[i] = abs((err[:,n_refs+i])/(ref_0-ref_1))  
    if (sp and (n_refs!=0)):
        print('\n',file_path[1+file_path.rfind('/'):])
        f,ax = plt.subplots(figsize=[6.4*2/3,4.8*2/3])
        for i in range(n_signals):
            ax.errorbar(xdat,ndat[i],yerr=errdat[i],fmt='.',ls=':',label='ms=-1')
        ax.grid()
        ax.set_xlabel(xText,fontsize=12)
        ax.set_ylabel(r'normalized signal',fontsize=12)
        ax.set_title(titleText,fontsize=12)
        if not(labels is None):
            plt.legend(labels,loc=0)
        plt.tight_layout()
        plt.show()
    return xdat,ndat,errdat



# default function to fit a gaussian(s) function to the pulsed_esr sequence
def esr_normalized(file_path,p0=None,doFit=True,nG=1,mov_avg_windowSize=0, retRes=False,n_signals=1,n_refs=1,skRows=globalskRows,sp=False,xfact=1e-9,xText='freq [GHz]',equidist=False,labels=None,ignore_signals=[]):
    print(file_path[1+file_path.rfind('/'):])
    xdata,ndat,nerr = signal_normalized(file_path,skRows=globalskRows,mov_avg_windowSize=mov_avg_windowSize,n_signals=n_signals,n_refs=n_refs,sp=sp,xfact=xfact,titleText='',xText='',labels=labels)
    fit = []
    f,ax = plt.subplots(figsize=[6.4*2/3,4.8*2/3])
    for i in range(len(ndat)):
        if i in ignore_signals:
            continue
        p00 = p0
        ydata,yderr = ndat[i],nerr[i] 
        ax.errorbar(xdata,ydata,yerr=yderr,fmt='.',ls=':')
        if doFit:
            xx = np.linspace(xdata[0], xdata[-1]);
            if nG==1:
                if p00 is None:
                    popt, perr, r2, *optVar = fit_gaussian(xdata, -ydata, yderr=yderr, p0=None,retRes=retRes)
                    popt = np.array(popt) * [-1, -1, 1, 1]
                else:
                    popt, perr, r2, *optVar = fit_func(funcGauss, xdata, ydata, p00,yderr=yderr,retRes=retRes)
                plt.plot(xx, funcGauss(xx, *popt), lw=2)
                print('{:<7s}{:>9s}{:>12s}{:>11s}{:>8s}'.format('', 'offset', 'amplitude', 'center', 'width'))
                print('{:<7s}{:>9.3f}{:>12.3f}{:>11.3f}{:>8.3f}'.format('value',abs(popt[0]),abs(popt[1]),popt[2],popt[3]))
                print('{:<7s}{:>9.3f}{:>12.3f}{:>11.3f}{:>8.3f}'.format('error',perr[0],perr[1],perr[2],perr[3]))
                if n_refs==0:
                    print('R² = %.4f'% (r2), '\ncontrast =', '%4.2f'% (-popt[1]/popt[0]*100), '%')
                else:
                    print('R² = %.4f'% (r2), '\ncontrast =', '%4.2f'% (-popt[1]*100), '%')
            elif nG==2:
                popt, perr, r2, *optVar = fit_func(func2Gauss, xdata, ydata, p00,yderr=yderr,retRes=retRes)
                plt.plot(xx, func2Gauss(xx, *popt), lw=2)
                print('{:<7s}{:>9s}{:>13s}{:>12s}{:>9s}{:>13s}{:>12s}{:>9s}'.format('', 'offset', 'amplitude1', 'center1', 'width1', 'amplitude2', 'center2', 'width2'))
                print('{:<7s}{:>9.3f}{:>13.3f}{:>12.3f}{:>9.3f}{:>13.3f}{:>12.3f}{:>9.3f}'.format('value',abs(popt[0]),abs(popt[1]),popt[2],popt[3],abs(popt[4]),popt[5],popt[6]))
                print('{:<7s}{:>9.3f}{:>13.3f}{:>12.3f}{:>9.3f}{:>13.3f}{:>12.3f}{:>9.3f}'.format('error',perr[0],perr[1],perr[2],perr[3],perr[4],perr[5],perr[6]))
                if n_refs==0:
                    print('R² = %.4f'% (r2), '\ncontrast =', '{:>7.2f}{:>7.2f}'.format((-popt[1]/popt[0]*100),(-popt[4]/popt[0]*100)))
                else:
                    print('R² = %.4f'% (r2), '\ncontrast =', '{:>7.2f}{:>7.2f}'.format((-popt[1]*100),(-popt[4]*100)))
            elif nG==3:
                if equidist:
                    if p00 is None:
                        p00 = estimators_gaussian(xdata,ydata)
                        p00 = np.concatenate([p00,[p00[1]/2,p00[1]/4]])
                    popt_aux, perr_aux, r2, *optVar = fit_func(func3GaussN14, xdata, ydata, p00,yderr=yderr,retRes=retRes)
                    popt = np.concatenate([popt_aux[:-1],[popt_aux[2]+2.15e6,popt_aux[3],popt_aux[-1],popt_aux[2]+2*2.15e6,popt_aux[3]]])
                    perr = np.concatenate([perr_aux[:-1],[perr_aux[2]+2.15e6,perr_aux[3],perr_aux[-1],perr_aux[2]+2*2.15e6,perr_aux[3]]])
                else:
                    popt, perr, r2, *optVar = fit_func(func3Gauss, xdata, ydata, p00,yderr=yderr,retRes=retRes)
                plt.plot(xx, func3Gauss(xx, *popt), lw=2)
                print('{:<7s}{:>9s}{:>13s}{:>12s}{:>9s}{:>13s}{:>12s}{:>9s}{:>13s}{:>12s}{:>9s}'.format('', 'offset', 'amplitude1', 'center1', 'width1', 'amplitude2', 'center2', 'width2', 'amplitude3', 'center3', 'width3'))
                print('{:<7s}{:>9.3f}{:>13.3f}{:>12.3f}{:>9.3f}{:>13.3f}{:>12.3f}{:>9.3f}{:>13.3f}{:>12.3f}{:>9.3f}'.format('value',abs(popt[0]),abs(popt[1]),popt[2],popt[3],abs(popt[4]),popt[5],popt[6],abs(popt[7]),popt[8],popt[9]))
                print('{:<7s}{:>9.3f}{:>13.3f}{:>12.3f}{:>9.3f}{:>13.3f}{:>12.3f}{:>9.3f}{:>13.3f}{:>12.3f}{:>9.3f}'.format('error',perr[0],perr[1],perr[2],perr[3],perr[4],perr[5],perr[6],perr[7],perr[8],perr[9]))
                if n_refs==0:
                    print('R² = %.4f'% (r2), '\ncontrast =', '{:>7.2f}{:>7.2f}{:>7.2f}'.format((-popt[1]/popt[0]*100),(-popt[4]/popt[0]*100),(-popt[7]/popt[0]*100)))
                else:
                    print('R² = %.4f'% (r2), '\ncontrast =', '{:>7.2f}{:>7.2f}{:>7.2f}'.format((-popt[1]*100),(-popt[4]*100),(-popt[7]*100)))
            else:
                print('Number of gaussians (nG) must be 1, 2 or 3')
            if retRes:
                fit.append([popt, perr, r2, optVar])
            else:
                fit.append([popt, perr, r2])
    ax.grid()
    ax.set_xlabel(xText,fontsize=12)
    ax.set_ylabel(r'normalized signal',fontsize=12)
    #ax.set_title(file_path[1+file_path.rfind('/'):],fontsize=12)
    if not(labels is None):
        plt.legend(labels,loc=0)
    plt.tight_layout()
    plt.show()
    if doFit:
        return fit
    else:
        return xdata,ndat,nerr
    

# default function to fit a sine function to the rabi sequence
def rabi_normalized(file_path,p0=None,doFit=True,decay=False,extra_phase=False,mov_avg_windowSize=0, retRes=False,n_signals=1,n_refs=1,skRows=globalskRows,sp=False,xfact=1e6,xText='MW time ($\mu$s)',labels=None,ignore_signals=[]):
    print(file_path[1+file_path.rfind('/'):])
    xdata,ndat,nerr = signal_normalized(file_path,skRows=globalskRows,mov_avg_windowSize=mov_avg_windowSize,n_signals=n_signals,n_refs=n_refs,sp=sp,xfact=xfact,titleText='',xText=xText,labels=labels)
    fit = []
    f,ax = plt.subplots(figsize=[6.4*2/3,4.8*2/3])
    for i in range(len(ndat)):
        if i in ignore_signals:
            continue
        p00 = p0
        ydata,yderr = ndat[i],nerr[i]    
        ax.errorbar(xdata,ydata,yerr=yderr,fmt='.',ls=':')
        if doFit:
            ## based on function lens.analysisFunctions.rabiAnalysis
            if p00 is None:
                xfft, yfft, fft_estims = fftaux(xdata, ydata, sP=False, return_estim=True)
                p00 = np.array([np.mean(ydata), ydata.max()-np.mean(ydata), 1/fft_estims[2]/2, fft_estims[3]])
                print('p00 =',p00)
                if decay:
                    p00 = np.concatenate([p00,[xdata.max()]])
            #function to fit
            if decay:
                def nutn(x,y0,a,t,ph,tau):
                    return y0+a*np.cos(np.pi*x/t-ph)*np.exp(-x/tau)
            else:
                def nutn(x,y0,a,t,ph):
                    return y0+a*np.cos(np.pi*x/t-ph)
            popt, perr, r2, *optVar = fit_func(nutn, xdata, ydata, p00, yderr=yderr, retRes=retRes)
            # p00,popt = [offset, amplitude, pi-pulse length, phase (, decay_time)]

            #For the first minimum:
            tt=popt[2]
            pha=popt[3]
            if extra_phase:
                xmin=tt*(pha/np.pi) # because of the phase, it sometimes is without the one
                xzero=tt*(-1/2+pha/np.pi) # because of the phase, it sometimes is with a possitive 1/2
            else:
                xmin=tt*(1+pha/np.pi)
                xzero=tt*(1/2+pha/np.pi)

            xx=np.linspace(xdata[0],xdata[-1],100);
            ax.plot(xx,nutn(xx,*popt),lw=2)
            ax.axvline(xmin,ls='--',color='red')
            ax.axvline(xzero,ls='--',color='red')
            ax.axhline(popt[0],ls='-',color='black')
            #
            print('1st minimum at x = %2.5f'% (xmin),'us')
            print('1st zero at    x = %2.5f'% (xzero),'us')
            if decay:
                print('{:<7s}{:>8s}{:>12s}{:>11s}{:>8s}{:>8s}'.format('', 'offset', 'amplitude', 'pi-pulse', 'phase', 'decay'))
                print('{:<7s}{:>8.4f}{:>12.3f}{:>11.5f}{:>8.3f}{:>8.2f}'.format('value',*popt))
                print('{:<7s}{:>8.4f}{:>12.3f}{:>11.5f}{:>8.3f}{:>8.2f}'.format('error',*perr))
            else:
                print('{:<7s}{:>8s}{:>12s}{:>11s}{:>8s}'.format('', 'offset', 'amplitude', 'pi-pulse', 'phase'))
                print('{:<7s}{:>8.4f}{:>12.3f}{:>11.5f}{:>8.3f}'.format('value',*popt))
                print('{:<7s}{:>8.4f}{:>12.3f}{:>11.5f}{:>8.3f}'.format('error',*perr))
            if n_refs==0:
                print('R² = %.4f'% (r2), '\nRabi freq = %2.4f'% (1/(2*popt[2])),'MHz',
                      '\ncontrast =', '%4.2f'% (2*popt[1]/(popt[0]+popt[1])*100), '%')
            else:
                print('R² = %.4f'% (r2), '\nRabi freq = %2.4f'% (1/(2*popt[2])),'MHz',
                      '\ncontrast =', '%4.2f'% (2*popt[1]*100), '%')
            if retRes:
                fit.append([popt, perr, r2, optVar])
            else:
                fit.append([popt, perr, r2])  
    ax.grid()
    ax.set_xlabel(xText,fontsize=12)
    ax.set_ylabel(r'normalized signal',fontsize=12)
    #ax.set_title(file_path[1+file_path.rfind('/'):],fontsize=12)
    if not(labels is None):
        plt.legend(labels,loc=0)
    plt.tight_layout()
    plt.show()
    if doFit:      
        return fit	
    else:
        return xdata,ndat,nerr


def ramsey_normalized(file_path,mwFreq=0,p0=None,doFit=True,nG=1,mov_avg_windowSize=0,n_signals=1,n_refs=1,skRows=globalskRows,sp=True,xfact=1e6,xText='tau [µs]',equidist=False,labels=None,splittings=[], nuphi=7.5, plot=False, return_FFT=False, return_DATA=False,retRes=False,add_zeros=1,ignore_signals=[]):
    print(file_path[1+file_path.rfind('/'):])
    xdata,ndat,nerr = signal_normalized(file_path,skRows=globalskRows,mov_avg_windowSize=mov_avg_windowSize,n_signals=n_signals,n_refs=n_refs,sp=sp,xfact=xfact,titleText='',xText='',labels=labels)
    if return_DATA:
        return xdata,ndat,nerr
    fit = []
    fft_list = [] # for return_FFT = True
    f,ax = plt.subplots(figsize=[6.4*2/3,4.8*2/3])
    for i in range(len(ndat)):
        if i in ignore_signals:
            continue
        p00 = p0
        print(p00)
        ydata,yderr = ndat[i],nerr[i] 
        freq, ampFT, sine_estimates = fftaux(xdata, ydata, sP=False, return_estim=True,add0s=add_zeros)
        if return_FFT:
            fft_list.append( [freq, ampFT] )
        if sp:
            ax.plot(freq,ampFT,'.:')
            ax.set_xlim([0, max(freq)])
            ax.set_xlabel(r'MW freq (MHz)',fontsize=22)
            ax.set_ylabel(r'Fluorescence intensity (kcps)',fontsize=18)
        if doFit:
            if p00 is None:
                # estimate initial parameters; p01 = [offset, amplitude, peak frequency, width]
                p00 = np.array([np.mean(ampFT), np.max(ampFT)-np.mean(ampFT), sine_estimates[2], 0.1])
                print(p00)
            print(p00)
            if nG==2:
                funcTemp=func2Lorentz
                p00=np.concatenate((p00,[p00[1], p00[2]+splittings[0], p00[3]]))
            elif nG==3:
                funcTemp=func3Lorentz
                p00=np.concatenate((p00,[p00[1], p00[2]+splittings[0], p00[3]],
                    [p00[1],p00[2]+splittings[0]+splittings[1], p00[3]]))
            elif nG==4:
                funcTemp=func4Lorentz
                p00=np.concatenate((p00,[p00[1], p00[2]+splittings[0], p00[3]],
                    [p00[1], p00[2]+splittings[0]+splittings[1], p00[3]],
                    [p00[1], p00[2]+splittings[0]+splittings[1]+splittings[2], p00[3]]))
            elif nG==6:
                funcTemp=func6Lorentz
                p00=np.concatenate((p00,[p00[1], p00[2]+splittings[0], p00[3]],
                    [p00[1], p00[2]+splittings[0]+splittings[1], p00[3]],
                    [p00[1], p00[2]+splittings[0]+splittings[1]+splittings[2], p00[3]],
                    [p00[1], p00[2]+splittings[0]+splittings[1]+splittings[2]+splittings[3], p00[3]],
                    [p00[1], p00[2]+splittings[0]+splittings[1]+splittings[2]+splittings[3]+splittings[4], p00[3]]))
            else:
                funcTemp=funcLorentz

            #fitting nG lorentzians:
            popt, perr, r2, *optVar  = fit_func(funcTemp,freq,ampFT,p00,retRes=retRes)
            if sp:
                if add_zeros:
                    ax.plot(freq,funcTemp(freq,*popt))
                else:
                    xx=np.linspace(0,freq.max(),1000)
                    ax.plot(xx,funcTemp(xx,*popt))

                text1='peaks in:'
                text2='MW frequencies:'
                for i in range(nG):
                    text1=text1+'\n'+str(popt[3*i+2])+' ± '+str(perr[3*i+2])+' MHz'
                    text2=text2+'\n'+str((mwFreq*1e3 - nuphi + popt[3*i+2])/1e3)+' ± '+str((perr[3*i+2])/1e3)+' GHz'
                print(text1)
                print(text2)

            if retRes:
                fit.append([ popt,perr,r2,optVar])
            else:
                fit.append([ popt,perr,r2])
        if sp:
            plt.show()
    if return_FFT:
        return fft_list
    if doFit:
        return fit




