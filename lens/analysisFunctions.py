# This is for functions that you use repeatedly during the analysis of the data
# content with description                                                      -(line):
# movAvgWeight (moving weighted-average)                                        -31
# fit_func (fit a specified function and return fitted parameters,errors & R²)  -67
# normalize (normalize the signal obtainted by the DD sequence)                 -89
# esr (default function to fit a gaussian to the pulsed_esr sequence)           -136
# find_nearest (function to find the nearest value in array (and its position)) -205
# ramsey (function to analyse ramsey measurement:fft and fit lorentzians)       -213
# ramsey_auto (same as previous but with estimators                             -328
# rabi (function to find the pi pulse fitting a sine^2 to a rabi sequence data) -427
# fftaux (function to calculate and plot the FFT of a signal)                   -504
# funcGauss,func2Gauss,func3Gauss (Gaussians)                                   -535:537:539
# funcLorentz,func2Lorentz,func3Lorentz,func4Lorentz,func6Lorentz (Lorentzians) -541:543:545:547:549
# weightAvg (function to calculate the weighted-average or array (with errors)) -555
# fit_gaussian (fit of a 1D gaussian distribution, with estimators)             -567
# funcCos (y0+a*np.cos(2*np.pi*nu*x-ph))                                        -578
# fit_cos (fit of a 1D cosine, with estimators)                                 -580
# fit_line (linear fit y=m*(x-x0), with estimators)                             -588





from lens.openDat import openDatFile

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import lens.fit_testing.lorentzian_fit_methods as lfm
from scipy.odr import ODR, Model, Data, RealData

#globalskRows = 6 #LENS
globalskRows = 0 #QEG

def movAvgWeight(ydata,yderr,windowSize):
    # This function calculates the moving average of an array including the errors
    # The function uses the definitions of weighted averages and standard error of the weighted averages given by:
    # (1) f-Robinson_Data reduction and error analysis for the physical sciences   pp 56-58
    # (2) Hugh D. Young - Statistical treatment of experimental Data                       pp 95,108
    # The function returns a tuple where the first element is the moving weighted-average and the second element
    #   is the error asociated with the average done in that specific window
    # NOTE1: the 'windowSize' is actually the diameter of the window
    # NOTE2: If you don't need to weight, you should use "pandas.rolling_mean( ydata,windowSize,center=True)"

    if (windowSize%2)!=0:
        print('Error: windowSize must be an even integer')
        return -1
    if ydata.shape!=yderr.shape:
        print('Error: ydata and yderr must be of the same dimension')
        return -1
    winRad = int(windowSize/2)
    result = np.zeros(len(ydata))
    resErr = np.zeros(len(yderr))

    for i in range(len(ydata)):
        if ((i < winRad)|(i>len(ydata)-1-winRad)):
            result[i] = np.nan # because we don't have enough points to perform an average
            resErr[i] = np.nan
        else:
            yd = ydata[i-winRad:i+winRad+1] # it takes 2*winRad + 1 values of the array around the i-th position
            ye = yderr[i-winRad:i+winRad+1]
            result[i] = np.nansum(yd/(ye**2))/np.nansum(1/(ye**2))
            #resErr[i] = np.sqrt( np.nansum(yd**2/(ye**2))/np.nansum(1/(ye**2))-result[i]**2 )*(1/np.sqrt(windowSize))
            resErr[i] = np.sqrt( np.nansum(yd**2/(ye**2))/np.nansum(1/(ye**2))-result[i]**2 )*(np.sqrt(windowSize+1)/np.sqrt(windowSize))
            # N = windowSize+1 => N-1 = windowSize
    return (result,resErr)




def fit_func(func,xdata,ydata,p0,yderr=None,retChi=False,constr=0,retRes=False):
    if constr==0:
        constr=(-np.inf,np.inf)
    popt, pcov = curve_fit(func, xdata, ydata, p0,sigma=yderr,bounds=constr)
    #popt, pcov = curve_fit(func, xdata, ydata, [50,230],sigma=yderr)
    perr = np.sqrt(np.diag(pcov))
    #calculation of R²
    residuals = ydata- func(xdata, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata-np.mean(ydata))**2)
    rSquared = 1 - (ss_res / ss_tot)
    if retRes:
        return (popt,perr,rSquared,[xdata,residuals])
    if (yderr!=[])&retChi:
        #calculation of χ²  -  see Bevington-Robinson_Data reduction... pp194,195
        xSquared = np.sum(residuals**2/yderr**2)
        return (popt,perr,rSquared,xSquared)
    return (popt,perr,rSquared)


def fit_func_odr(func_odr,xdata,ydata,p0,xderr=None,yderr=None,return_output=False,maxit=None):
    """
    Function to do a fit with the Orthogonal Distance Regression method of scipy (allows error in X and Y)
    Note:
    func_odr is a function fcn(beta, x) –> y (example: fcn(beta, x) -> beta[0]*x + beta[1])
    """
    data = RealData(xdata, ydata, sx=xderr, sy=yderr)
    model = Model(func_odr)
    odr = ODR(data, model, p0, maxit=maxit)
    odr.set_job(fit_type=0)
    output = odr.run()
    if return_output:
        return output
    else:
        popt = output.beta
        perr = np.maximum(np.sqrt(np.diag(output.cov_beta)),output.sd_beta)
        #calculation of R²
        residuals = ydata- func_odr(output.beta, xdata)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((ydata-np.mean(ydata))**2)
        rSquared = 1 - (ss_res / ss_tot)
        return (popt,perr,rSquared)


def normalize(data,plot=False,plotTitle='',plotTexts='',returnAvgs=False,returnAvgsWithErr=False,const1=2,labelX=r'T=2t$_1$ [$\mu$s]',twoRef=False,alph=0.1): #,savePlot=False):
    # function to normalize the signal obtained by the sequence: "dynamical_decoupling_with_RF_2bis"
    # If the argument 'plot' is True then it shows the plot.
    # The function returns three arrays: (2*t1 [in us], normalized signal, error of the norm sign)
    if twoRef:
        col1=data.shape[1]-8
    else:
        col1=data.shape[1]-6
    xx=const1*data[:,0]*1e6
    if plot:
        plt.errorbar(xx,data[:,col1+4],yerr=data[:,col1+5],fmt='o',ls='-',alpha=alph)
        plt.plot([xx.min(),xx.max()],[data[:,col1+4].mean(),data[:,col1+4].mean()],ls='-',color='blue',label='ms=0')
        plt.errorbar(xx,data[:,col1+2],yerr=data[:,col1+3],fmt='o',ls='-',label='signal')
        plt.errorbar(xx,data[:,col1],yerr=data[:,col1+1],fmt='o',ls='-',alpha=alph)
        plt.plot([xx.min(),xx.max()],[data[:,col1].mean(),data[:,col1].mean()],ls='-',color='red',label='ms=-1')
        if twoRef:
            plt.errorbar(xx,data[:,col1+6],yerr=data[:,col1+7],fmt='o',ls='-',alpha=alph)
            plt.plot([xx.min(),xx.max()],[data[:,col1+6].mean(),data[:,col1+6].mean()],ls='-',color='purple',label='ms=0')
        plt.xlabel(labelX,fontsize=22)
        plt.ylabel(r'[cps]',fontsize=22)
        #plt.legend(loc=5)#'best')
        plt.title(plotTitle,fontsize=20)
        plt.text(xx.min(),data[:,col1+4].max(),plotTexts,fontsize=12)
        plt.tight_layout()
        #if savePlot:
        #    plt.savefig('./2017-11-23_'+plotTitle+plotTexts+'.png',format='png',dpi=100)
        plt.show()


    if twoRef:
        array2return = np.array([xx,100*(data[:,col1+2]-np.mean(data[:,col1]))/(np.mean(data[:,col1+6])-np.mean(data[:,col1])),100*(data[:,col1+1])/(np.mean(data[:,col1])-np.mean(data[:,col1+6])) ])
        if returnAvgsWithErr:
            return (np.mean(data[:,col1]),np.std(data[:,col1]),np.mean(data[:,col1+4]),np.std(data[:,col1+4]),np.mean(data[:,col1+6]),np.std(data[:,col1+6]), array2return )
        if returnAvgs:
            return (np.mean(data[:,col1]), np.mean(data[:,col1+4]), np.mean(data[:,col1+6]),  array2return)
        return array2return
    else:
        array2return = np.array([xx,100*(data[:,col1+2]-np.mean(data[:,col1+4]))/(np.mean(data[:,col1])-np.mean(data[:,col1+4])),100*(data[:,col1+1])/(np.mean(data[:,col1+4])-np.mean(data[:,col1])) ])
        if returnAvgsWithErr:
            return (np.mean(data[:,col1]),np.std(data[:,col1]),np.mean(data[:,col1+4]),np.std(data[:,col1+4]), array2return )
        if returnAvgs:
            return (np.mean(data[:,col1]), np.mean(data[:,col1+4]),  array2return)
        return array2return


def normalize_spinMap(file_path, order=['msm1', 'ms0', 'msp1', 'ref1', 'ref0'], measType='avg', ms4rabi=[0,-1],
                      invertNorm=False, xlabel='no-meaning variable',xfact=1e6,skRows=globalskRows):
    data = openDatFile(file_path,skRows=skRows)

    order.reverse()
    xdat = np.array(data[:,0]) * xfact
    yrefms0  = data[:,-2*order.index('ref0')-2]
    yrefms0E = data[:,-2*order.index('ref0')-1]
    yrefms1  = data[:,-2*order.index('ref1')-2]
    yrefms1E = data[:,-2*order.index('ref1')-1]
    if 'msp1' in order:
        ymsp1  = data[:,-2*order.index('msp1')-2]
        ymsp1E = data[:,-2*order.index('msp1')-1]
    if 'ms0' in order:
        yms0  = data[:,-2*order.index('ms0')-2]
        yms0E = data[:,-2*order.index('ms0')-1]
    if 'msm1' in order:
        ymsm1  = data[:,-2*order.index('msm1')-2]
        ymsm1E = data[:,-2*order.index('msm1')-1]
    #
    refms0, refms0E = weightAvg(yrefms0, yrefms0E)
    refms1, refms1E = weightAvg(yrefms1, yrefms1E)
    contrastRef = (refms0 - refms1) / refms0
    contrastRefE = refms1E / refms0 + refms0E * refms1 / refms0 ** 2

    if measType == 'avg':
        # make plot
        f,(ax1,ax2) = plt.subplots(1,2,figsize=[14,4])
        if 'msp1' in order:
            msp1,msp1E = weightAvg(ymsp1,ymsp1E)
            popp1 = (refms0-msp1)/(refms0-refms1)
            if invertNorm: popp1 = 1 - popp1
            contrastp1 = (refms0-msp1)/refms0
            popp1E = np.sqrt(refms0E**2+msp1E**2)/(refms0-refms1) + \
                     np.sqrt(refms0E**2+refms1E**2)*(refms0-msp1)/(refms0-refms1)**2
            contrastp1E = msp1E/refms0 + refms0E*msp1/refms0**2
            ax1.errorbar(xdat,ymsp1,yerr=ymsp1E,label='$m_S = +1$',fmt='o',ls='-')
            ax2.errorbar(0,msp1,yerr=msp1E,label='mean $m_S = +1$',fmt='o')
            print('mean mS=+1', int(round(ymsp1.mean())), '±', int(round(ymsp1.std())), 'cps')
        if 'ms0' in order:
            ms0,ms0E = weightAvg(yms0,yms0E)
            pop0 = (refms0-ms0)/(refms0-refms1)
            if invertNorm: pop0 = 1 - pop0
            contrast0 = (refms0-ms0)/refms0
            pop0E = np.sqrt(refms0E**2+ms0E**2)/(refms0-refms1) + \
                   np.sqrt(refms0E**2+refms1E**2)*(refms0-ms0)/(refms0-refms1)**2
            contrast0E = ms0E/refms0 + refms0E*ms0/refms0**2
            ax1.errorbar(xdat,yms0,yerr=yms0E,label='$m_S = 0$',fmt='o',ls='-')
            ax2.errorbar(0,ms0,yerr=ms0E,label='mean $m_S = 0$',fmt='o')
            print('mean mS=0', int(round(yms0.mean())), '±', int(round(yms0.std())), 'cps')
        if 'msm1' in order:
            msm1,msm1E = weightAvg(ymsm1,ymsm1E)
            popm1 = (refms0-msm1)/(refms0-refms1)
            if invertNorm: popm1 = 1 - popm1
            contrastm1 = (refms0-msm1)/refms0
            popm1E = np.sqrt(refms0E**2+msm1E**2)/(refms0-refms1) + \
                     np.sqrt(refms0E**2+refms1E**2)*(refms0-msm1)/(refms0-refms1)**2
            contrastm1E = msm1E/refms0 + refms0E*msm1/refms0**2
            ax1.errorbar(xdat,ymsm1,yerr=ymsm1E,label='$m_S = -1$',fmt='o',ls='-')
            ax2.errorbar(0,msm1,yerr=msm1E,label='mean $m_S = -1$',fmt='o')
            print('mean mS=-1', int(round(ymsm1.mean())), '±', int(round(ymsm1.std())), 'cps')
        ax1.errorbar(xdat,yrefms1,yerr=yrefms1E,label='ref $m_S = +1$',fmt='o',ls='-')
        ax1.errorbar(xdat,yrefms0,yerr=yrefms0E,label='ref $m_S = 0$',fmt='o',ls='-')
        #ax1.set_ylim(6.0e4, 9.0e4)
        ax1.legend(loc='lower right', framealpha=0.5)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('fluorescence intensity [cps]')
        ax1.set_title(file_path[-71:-10])
        #
        ax2.errorbar(0,refms1,yerr=refms1E,label='mean ref',fmt='o')
        ax2.errorbar(0,refms0,yerr=refms0E,label='mean $m_I = 0$',fmt='o')
        #ax2.set_ylim(6.5e4, 9e4)
        ax2.legend(loc='lower right')
        ax2.set_xlabel("x-axis doesn't exist")
        ax2.set_ylabel('fluorescence intensity [cps]')
        print('mean ref mS= +1', int(round(yrefms1.mean())), '±', int(round(yrefms1.std())), 'cps')
        print('mean ref mS= 0', int(round(yrefms0.mean())), '±', int(round(yrefms0.std())), 'cps')
        plt.show()

        allPopul  = []
        allPopulE = []
        if 'msp1' in order:
            print('mS = +1 population', round(popp1*100,2), '±', round(popp1E*100,2), '%')
            print('mS = +1 contrast', round(contrastp1*100,2), '±', round(contrastp1E*100,2), '%')
            allPopul.append(popp1)
            allPopulE.append(popp1E)
        if 'ms0' in order:
            print('mS = 0 population', round(pop0*100,2), '±', round(pop0E*100,2), '%')
            print('mS = 0 contrast', round(contrast0*100,2), '±', round(contrast0E*100,2), '%')
            allPopul.append(pop0)
            allPopulE.append(pop0E)
        if 'msm1' in order:
            print('mS = -1 population', round(popm1*100,2), '±', round(popm1E*100,2), '%')
            print('mS = -1 contrast', round(contrastm1*100,2), '±', round(contrastm1E*100,2), '%')
            allPopul.append(popm1)
            allPopulE.append(popm1E)
        allPopul = np.array(allPopul)
        allPopulE = np.array(allPopulE)
        print('sum population', round(allPopul.sum()*100,2), '±', round(np.sqrt((allPopulE**2).sum())*100,2), '%')
        print('\nref mS = 0 <-> +1 contrast', round(contrastRef*100,2), '±', round(contrastRefE*100,2))
        return allPopul, allPopulE

    elif measType == 'justNorm':
        # make plot
        f,(ax1,ax2) = plt.subplots(1,2,figsize=[14,4])
        if 'msp1' in order:
            popp1 = (refms0-ymsp1)/(refms0-refms1)
            if invertNorm: popp1 = 1 - popp1
            popp1E = np.sqrt(refms0E**2+ymsp1E**2)/(refms0-refms1) + \
                     np.sqrt(refms0E**2+refms1E**2)*(refms0-ymsp1)/(refms0-refms1)**2
            ax1.errorbar(xdat,ymsp1,yerr=ymsp1E,label='$m_S = +1$',fmt='o',ls='-')
            ax2.errorbar(xdat,popp1,yerr=popp1E,label='$m_S = +1$',fmt='o',ls='-')
        if 'ms0' in order:
            pop0 = (refms0-yms0)/(refms0-refms1)
            if invertNorm: pop0 = 1 - pop0
            pop0E = np.sqrt(refms0E**2+yms0E**2)/(refms0-refms1) + \
                   np.sqrt(refms0E**2+refms1E**2)*(refms0-yms0)/(refms0-refms1)**2
            ax1.errorbar(xdat,yms0,yerr=yms0E,label='$m_S = 0$',fmt='o',ls='-')
            ax2.errorbar(xdat,pop0,yerr=pop0E,label='$m_S = 0$',fmt='o',ls='-')
        if 'msm1' in order:
            popm1 = (refms0-ymsm1)/(refms0-refms1)
            if invertNorm: popm1 = 1 - popm1
            popm1E = np.sqrt(refms0E**2+ymsm1E**2)/(refms0-refms1) + \
                     np.sqrt(refms0E**2+refms1E**2)*(refms0-ymsm1)/(refms0-refms1)**2
            ax1.errorbar(xdat,ymsm1,yerr=ymsm1E,label='$m_S = -1$',fmt='o',ls='-')
            ax2.errorbar(xdat,popm1,yerr=popm1E,label='$m_S = -1$',fmt='o',ls='-')
        ax1.errorbar(xdat,yrefms1,yerr=yrefms1E,label='ref $m_S = +1$',fmt='o',ls='-')
        ax1.errorbar(xdat,yrefms0,yerr=yrefms0E,label='ref $m_S = 0$',fmt='o',ls='-')
        #ax1.set_ylim(6.0e4, 9.0e4)
        ax1.legend(loc='lower right', framealpha=0.5)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('fluorescence intensity [cps]')
        ax1.set_title(file_path[-71:-10])
        #ax2.set_ylim(6.5e4, 9e4)
        ax2.legend(loc='lower right')
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel('Population')
        print('mean ref mS= +1:', int(round(yrefms1.mean())), '±', int(round(yrefms1.std())), 'cps')
        print('mean ref mS= 0:', int(round(yrefms0.mean())), '±', int(round(yrefms0.std())), 'cps')
        plt.show()

        allPopul  = []
        allPopulE = []
        if 'msp1' in order:
            allPopul.append(popp1)
            allPopulE.append(popp1E)
        if 'ms0' in order:
            allPopul.append(pop0)
            allPopulE.append(pop0E)
        if 'msm1' in order:
            allPopul.append(popm1)
            allPopulE.append(popm1E)
        allPopul = np.array(allPopul)
        allPopulE = np.array(allPopulE)
        print('\nref mS = 0 <-> +1 contrast', round(contrastRef*100,2), '±', round(contrastRefE*100,2))
        return allPopul, allPopulE, xdat

    elif measType == 'Rabi':
        plt.figure()
        if 'msp1' in order:
            plt.errorbar(xdat,ymsp1,yerr=ymsp1E,label='$m_S = +1$',fmt='o',ls='-')
        if 'ms0' in order:
            plt.errorbar(xdat,yms0,yerr=yms0E,label='$m_S = 0$',fmt='o',ls='-')
        if 'msm1' in order:
            plt.errorbar(xdat,ymsm1,yerr=ymsm1E,label='$m_S = -1$',fmt='o',ls='-')
        plt.errorbar(xdat,yrefms1,yerr=yrefms1E,label='ref $m_S = +1$',fmt='o',ls='-')
        plt.errorbar(xdat,yrefms0,yerr=yrefms0E,label='ref $m_S = 0$',fmt='o',ls='-')
        plt.legend(loc='lower right', framealpha=0.5)
        plt.xlabel(xlabel)
        plt.ylabel('fluorescence intensity [cps]')
        plt.title(file_path[-71:-10])
        plt.show()
        #
        allresults = []
        for msvalue in ms4rabi:
            if msvalue==1:
                ydata = (refms0-ymsp1)/(refms0-refms1)
                if invertNorm: ydata = 1 - ydata
                yderr = np.sqrt(refms0E**2+ymsp1E**2)/(refms0-refms1) + \
                        np.sqrt(refms0E**2+refms1E**2)*(refms0-ymsp1)/(refms0-refms1)**2
                #ydata,yderr = ymsp1, ymsp1E
            elif msvalue==0:
                ydata = (refms0-yms0)/(refms0-refms1)
                if invertNorm: ydata = 1 - ydata
                yderr = np.sqrt(refms0E**2+yms0E**2)/(refms0-refms1) + \
                        np.sqrt(refms0E**2+refms1E**2)*(refms0-yms0)/(refms0-refms1)**2
                #ydata,yderr = yms0, yms0E
            elif msvalue==-1:
                ydata = (refms0-ymsm1)/(refms0-refms1)
                if invertNorm: ydata = 1 - ydata
                yderr = np.sqrt(refms0E**2+ymsm1E**2)/(refms0-refms1) + \
                        np.sqrt(refms0E**2+refms1E**2)*(refms0-ymsm1)/(refms0-refms1)**2
                #ydata,yderr = ymsm1, ymsm1E
            else:
                print('ms4rabi must contain only +1, 0 and/or -1')
            popt, perr, r2 = rabiAnalysis(xdat, ydata,yderr, file=file_path,
                    xlab=r'RF time ($\mu$s)',ylab=r'Popul. $m_S='+str(msvalue)+'$')
            allresults.append([popt, perr, r2])
            allresults.extend([allPopul, allPopulE, xdat])
        print('\nref mS = 0 <-> +1 contrast', round(contrastRef*100,2), '±', round(contrastRefE*100,2))
        return allresults
    else:
        print('Choose correct measurement type: "avg" or "Rabi"!')



def esr(file_path,p0=None,doFit=True,nG=1,retRes=False,ref=False,nmr=False,equidist=False,skRows=globalskRows):
    # default function to fit a gaussian to the pulsed_esr sequence
    print('\n',file_path[-44:])
    data=openDatFile(file_path,skRows=skRows)
    if nmr:
        if ref:
            col1=data.shape[1]-6
            refdata=np.array([data[:,data.shape[1]-2],data[:,data.shape[1]-1]])
        else:
            col1=data.shape[1]-4
        data0 = np.array([data[:,0],data[:,col1],data[:,col1+1]])
        data1 = np.array([data[:,0],data[:,col1+2],data[:,col1+3]])
        data2 = (data0+data1)/2
        data2[2,:]=data2[2,:]/np.sqrt(2)
    else:
        if ref:
            col1=data.shape[1]-4
            refdata=np.array([data[:,data.shape[1]-2],data[:,data.shape[1]-1]])
        else:
            col1=data.shape[1]-2
        data2=np.array([data[:,0],data[:,col1],data[:,col1+1]])
    #
    xdata = (data2)[0,:]
    ydata = (data2)[1,:]
    yderr = (data2)[2,:]
    plt.figure()#figsize=[8,6])
    if ref:
        plt.errorbar(xdata,refdata[0,:],yerr=refdata[1,:],fmt='.',ls='',color='r')
    plt.errorbar(xdata,ydata,yerr=yderr,fmt='o',ls='',label='ms=-1',color='grey')

    if doFit:
        xx = np.linspace(xdata[0], xdata[-1]);
        if nG==1:
            if p0 is None:
                popt, perr, r2, *optVar = fit_gaussian(xdata, -ydata, yderr=yderr, p0=None,retRes=retRes)
                popt = np.array(popt) * [-1, -1, 1, 1]
            else:
                popt, perr, r2, *optVar = fit_func(funcGauss, xdata, ydata, p0,yderr=yderr,retRes=retRes)
            plt.plot(xx, funcGauss(xx, *popt), lw=2)
        elif nG==2:
            popt, perr, r2, *optVar = fit_func(func2Gauss, xdata, ydata, p0,yderr=yderr,retRes=retRes)
            plt.plot(xx, func2Gauss(xx, *popt), lw=2)
        elif nG==3:
            if equidist:
                if p0 is None:
                    p0 = estimators_gaussian(xdata,ydata)
                    p0 = np.concatenate([p0,[p0[1]/2,p0[1]/4]])
                popt_aux, perr_aux, r2, *optVar = fit_func(func3GaussN14, xdata, ydata, p0,yderr=yderr,retRes=retRes)
                popt = np.concatenate([popt_aux[:-1],[popt_aux[2]+2.15e6,popt_aux[3],popt_aux[-1],popt_aux[2]+2*2.15e6,popt_aux[3]]])
                perr = np.concatenate([perr_aux[:-1],[perr_aux[2]+2.15e6,perr_aux[3],perr_aux[-1],perr_aux[2]+2*2.15e6,perr_aux[3]]])
            else:
                popt, perr, r2, *optVar = fit_func(func3Gauss, xdata, ydata, p0,yderr=yderr,retRes=retRes)
            plt.plot(xx, func3Gauss(xx, *popt), lw=2)
        else:
            print('Number of gaussians (nG) must be 1, 2 or 3')
    plt.grid()
    plt.xlabel(r'MW freq (Hz)',fontsize=22)
    plt.ylabel(r'Fluorescence intensity (cps)',fontsize=18)
    plt.title(file_path[-44:])
    plt.show()
    if doFit:
        if nG==1:
            print('{:<7s}{:>9s}{:>12s}{:>11s}{:>8s}'.format('', 'offset', 'amplitude', 'center', 'width'))
            print('{:<7s}{:>9.0f}{:>12.0f}{:>11.3f}{:>8.3f}'.format('value',abs(popt[0]),abs(popt[1]),1e-6*popt[2],1e-6*popt[3]))
            print('{:<7s}{:>9.0f}{:>12.0f}{:>11.3f}{:>8.3f}'.format('error',perr[0],perr[1],1e-6*perr[2],1e-6*perr[3]))
            print('R² = %.4f'% (r2), '\ncontrast =', '%4.2f'% (-popt[1]/popt[0]*100), '%')
        elif nG==2:
            print('{:<7s}{:>9s}{:>13s}{:>12s}{:>9s}{:>13s}{:>12s}{:>9s}'.format('', 'offset', 'amplitude1', 'center1', 'width1', 'amplitude2', 'center2', 'width2'))
            print('{:<7s}{:>9.0f}{:>13.0f}{:>12.3f}{:>9.3f}{:>13.0f}{:>12.3f}{:>9.3f}'.format('value',abs(popt[0]),abs(popt[1]),1e-6*popt[2],1e-6*popt[3],abs(popt[4]),1e-6*popt[5],1e-6*popt[6]))
            print('{:<7s}{:>9.0f}{:>13.0f}{:>12.3f}{:>9.3f}{:>13.0f}{:>12.3f}{:>9.3f}'.format('error',perr[0],perr[1],1e-6*perr[2],1e-6*perr[3],perr[4],1e-6*perr[5],1e-6*perr[6]))
            print('R² = %.4f'% (r2), '\ncontrast =', '{:>7.2f}{:>7.2f}'.format((-popt[1]/popt[0]*100),(-popt[4]/popt[0]*100)))
        elif nG==3:
            print('{:<7s}{:>9s}{:>13s}{:>12s}{:>9s}{:>13s}{:>12s}{:>9s}{:>13s}{:>12s}{:>9s}'.format('', 'offset', 'amplitude1', 'center1', 'width1', 'amplitude2', 'center2', 'width2', 'amplitude3', 'center3', 'width3'))
            print('{:<7s}{:>9.0f}{:>13.0f}{:>12.3f}{:>9.3f}{:>13.0f}{:>12.3f}{:>9.3f}{:>13.0f}{:>12.3f}{:>9.3f}'.format('value',abs(popt[0]),abs(popt[1]),1e-6*popt[2],1e-6*popt[3],abs(popt[4]),1e-6*popt[5],1e-6*popt[6],abs(popt[7]),1e-6*popt[8],1e-6*popt[9]))
            print('{:<7s}{:>9.0f}{:>13.0f}{:>12.3f}{:>9.3f}{:>13.0f}{:>12.3f}{:>9.3f}{:>13.0f}{:>12.3f}{:>9.3f}'.format('error',perr[0],perr[1],1e-6*perr[2],1e-6*perr[3],perr[4],1e-6*perr[5],1e-6*perr[6],perr[7],1e-6*perr[8],1e-6*perr[9]))
            print('R² = %.4f'% (r2), '\ncontrast =', '{:>7.2f}{:>7.2f}{:>7.2f}'.format((-popt[1]/popt[0]*100),(-popt[4]/popt[0]*100),(-popt[7]/popt[0]*100)))
        if retRes:
            return popt, perr, r2, optVar
        else:
            return popt, perr, r2
    else:
        if ref:
            return xdata,ydata,yderr,refdata
        else:
            return xdata,ydata,yderr


def esr_fixedSplitting(file_path, add_params=None, ref=0, make_plot=True,skRows=globalskRows):
    # function to fit triple Lorentzian with fixed hyperfine splitting and shared dip widths to the pulsed_esr sequence
    data = openDatFile(file_path,skRows=skRows)
    col1 = data.shape[1] - 2*(ref+1)
    data2 = np.array([data[:, 0], data[:, col1], data[:, col1 + 1]])
    xdata = (data2)[0, :]
    ydata = (data2)[1, :]
    #yderr = (data2)[2, :]

    lor_fit = lfm.make_lorentziantriple_fit(xdata, ydata, lfm.estimate_lorentziantriple_N14, add_params=add_params)
    lor_model, params = lfm.make_lorentziantriple_model()
    xfit = np.linspace(xdata.min(), xdata.max(), 200)
    yfit = lor_model.eval(x=xfit, **lor_fit.best_values)
    if make_plot:
        plt.plot(xdata, ydata, 'bo:')
        plt.plot(xfit, yfit, '-r')
        plt.xlabel('MW freq (Hz)')
        plt.ylabel('counts (cps)')
        plt.title(file_path[len(file_path):])
        plt.show()
    return lor_fit


def calc_nuc_pol(lor_fit, which='center'):
    # function to calculate the polarization fraction of the N15 nuclear spin of an ESR measurement. Input requires
    # the fit result from the function 'esr_fixedSplitting'
    contrast_l = [lor_fit.result_str_dict['Contrast 0']['value'], lor_fit.result_str_dict['Contrast 0']['error']]
    contrast_c = [lor_fit.result_str_dict['Contrast 1']['value'], lor_fit.result_str_dict['Contrast 1']['error']]
    contrast_r = [lor_fit.result_str_dict['Contrast 2']['value'], lor_fit.result_str_dict['Contrast 2']['error']]
    contrast_l = np.round(np.array(contrast_l), 2)
    contrast_c = np.round(np.array(contrast_c), 2)
    contrast_r = np.round(np.array(contrast_r), 2)
    print('Contrast left  :', contrast_l[0], '±', contrast_l[1], '%')
    print('Contrast center:', contrast_c[0], '±', contrast_c[1], '%')
    print('Contrast right :', contrast_r[0], '±', contrast_r[1], '%')
    if which == 'left':
        pol_frac = contrast_l[0] / (contrast_l[0] + contrast_c[0] + contrast_r[0])
        pol_frac_err = contrast_l[1] / (contrast_l[0]+contrast_c[0]+contrast_r[0]) + \
            contrast_l[0]*(contrast_l[1]+contrast_c[1]+contrast_r[1]) / (contrast_l[0]+contrast_c[0]+contrast_r[0])**2
        print('\033[1m' + 'Pol. fraction left:', round(pol_frac*100, 2), '±', round(pol_frac_err*100, 2), '%' + '\033[0m')
    elif which == 'center':
        pol_frac = contrast_c[0] / (contrast_l[0] + contrast_c[0] + contrast_r[0])
        pol_frac_err = contrast_c[1] / (contrast_l[0]+contrast_c[0]+contrast_r[0]) + \
            contrast_c[0]*(contrast_l[1]+contrast_c[1]+contrast_r[1]) / (contrast_l[0]+contrast_c[0]+contrast_r[0])**2
        print('\033[1m' + 'Pol. fraction center:', round(pol_frac*100, 2), '±', round(pol_frac_err*100, 2), '%' + '\033[0m')
    elif which == 'right':
        pol_frac = contrast_r[0] / (contrast_l[0] + contrast_c[0] + contrast_r[0])
        pol_frac_err = np.sqrt((contrast_r[1] / (contrast_l[0]+contrast_c[0]+contrast_r[0]))**2 + \
            (contrast_r[0]*(contrast_l[1]+contrast_c[1]+contrast_r[1]) / (contrast_l[0]+contrast_c[0]+contrast_r[0])**2)**2)
        print('\033[1m' + 'Pol. fraction right:', round(pol_frac*100, 2), '±', round(pol_frac_err*100, 2), '%' + '\033[0m')
    else:
        print("ERROR: Choose for which dip to calculate the polarization: 'left', 'center' or 'right'.")
    return



def find_nearest(array,value):
    #function to find the nearest value in array (and its position):
    idx = (np.abs(array-value)).argmin()
    return array[idx],idx



def ramsey(fileName,dataType, p0,mwFreq,nG=2,nuphi=7.5,plot=False,retData=False,carbon=True,retRes=False,add_zeros=True,skRows=globalskRows):
    print(fileName[-45:])
    if ((dataType!=0)&(dataType!=1))&(dataType!=2):
        return 'Error: dataType should be 0 or 1 or 2 (0=signal  1=signal+ref+ref  2=signal+ref)'
    dataAll=openDatFile(fileName,skRows=skRows)
    ##reRead0:
    if dataType==0:
        col1=dataAll.shape[1]-2
        if plot:
            print('Signal mean value =', dataAll[:,col1].mean())
            plt.errorbar(dataAll[:,0],dataAll[:,col1],yerr=dataAll[:,col1+1],fmt='o',ls='-',label='signal')
            plt.show()
        data = np.array([dataAll[:,0]*1e6,dataAll[:,col1],dataAll[:,col1+1]])
        #data = reRead0(dataAll,True)
    ##reRead2:
    elif dataType==1:
        col1=dataAll.shape[1]-6
        if plot:
            print('Signal mean value =', dataAll[:,col1+2].mean())
            plt.errorbar(dataAll[:,0],dataAll[:,col1+4],yerr=dataAll[:,col1+5],fmt='o',ls='-',label='ms=0')
            plt.errorbar(dataAll[:,0],dataAll[:,col1+2],yerr=dataAll[:,col1+3],fmt='o',ls='-',label='signal')
            plt.errorbar(dataAll[:,0],dataAll[:,col1],yerr=dataAll[:,col1+1],fmt='o',ls='-',label='ms=-1')
            plt.show()
        data = np.array([dataAll[:,0]*1e6,dataAll[:,col1+2],dataAll[:,col1+3]])
        #data = reRead2(dataAll,True)
    elif dataType==2:
        col1=dataAll.shape[1]-4
        if plot:
            print('Signal mean value =', dataAll[:,col1+2].mean())
            plt.errorbar(dataAll[:,0],dataAll[:,col1+2],yerr=dataAll[:,col1+3],fmt='o',ls='-',label='signal')
            plt.errorbar(dataAll[:,0],dataAll[:,col1],yerr=dataAll[:,col1+1],fmt='o',ls='-',label='ms=0')
            plt.show()
        data = np.array([dataAll[:,0]*1e6,dataAll[:,col1+2]/dataAll[:,col1],0*dataAll[:,col1+3]])
        #data = reRead2(dataAll,True)
    #
    if ((len(p0)-1)/3)!=nG:
        return 'ERROR: p0 not compatible with '+str(nG)+' gaussians'

    dd=data[1,:]-data[1,:].mean() #remove background
    #add zeros to increase resolution
    if add_zeros:
        dd=np.concatenate((dd,np.zeros(len(dd)),np.zeros(len(dd)),np.zeros(len(dd)),np.zeros(len(dd))))
    dt=data[0,1]-data[0,0] #time step
    N=len(dd) #number of values

    ft=np.fft.fft(dd)*dt #calculate the fft
    freq = np.fft.fftfreq(N, dt) # generate freq array
    freq = freq[:int(N/2+1)] #take only half of the freq array (the positive part)

    ampFT=np.abs(ft[:int(N/2+1)]) #amplitude of the fft

    if retData:
        return freq,ampFT

    plt.figure()
    plt.plot(freq,ampFT,'.',ls='-')
    plt.xlim([0, max(freq)])
    plt.xlabel(r'MW freq (MHz)',fontsize=22)
    plt.ylabel(r'Fluorescence intensity (kcps)',fontsize=18)
    plt.title(fileName[-45:])

    if nG==1:
        funcTemp=funcLorentz
    elif nG==3:
        funcTemp=func3Lorentz
    elif nG==4:
        funcTemp=func4Lorentz
    elif nG==6:
        funcTemp=func6Lorentz
    else:
        funcTemp=func2Lorentz

    #fitting nG lorentzians:
    popt, perr, r2, *optVar  = fit_func(funcTemp,freq,ampFT,p0,retRes=retRes)
    if add_zeros:
        plt.plot(freq,funcTemp(freq,*popt))
    else:
        xx=np.linspace(0,freq.max(),1000)
        plt.plot(xx,funcTemp(xx,*popt))

    text1='peaks in'
    for i in range(nG):
        text1=text1+'\n'+str(popt[3*i+2])+' ± '+str(perr[3*i+2])+' MHz'
    print(text1)

    if carbon:
        if nG==3:
            print('Δν mI±1,0:',popt[8]-popt[5],popt[5]-popt[2],'MHz')
            print('Area of mI=+1/All areas = ',(popt[7])/(popt[1]+popt[4]+popt[7]))
            print('New MW Freq should be: ',(mwFreq*1e3 - nuphi + popt[[2,5,8]].mean())/1e3,'GHz (obtained as mean of 6 peaks)')
        elif nG==4:
            print('Δν mI±1/2:',-popt[2]+popt[5],-popt[8]+popt[11],'MHz')
            print('Δν mI+1,0:',(popt[8]+popt[11])/2-(popt[2]+popt[5])/2,'MHz')
            print('Area of mI=+1/All areas = ',(popt[7]+popt[10])/(popt[1]+popt[4]+popt[7]+popt[10]))
        elif nG==6:
            print('Δν mI±1/2:',-popt[2]+popt[5],-popt[8]+popt[11],-popt[14]+popt[17],'MHz')
            print('Δν mI±1,0:',(popt[8]+popt[11])/2-(popt[2]+popt[5])/2,(popt[8]+popt[11])/2-(popt[14]+popt[17])/2,'MHz')
            print('Area of mI=+1/All areas = ',(popt[13]+popt[16])/(popt[1]+popt[4]+popt[7]+popt[10]+popt[13]+popt[16]))
            print('New MW Freq should be: ',(mwFreq*1e3 - nuphi + popt[[2,5,8,11,14,17]].mean())/1e3,'GHz (obtained as mean of 6 peaks)')
        else:
            print('New MW Freq should be: ',(mwFreq*1e3 - nuphi + (popt[2]+popt[5])/2)/1e3,'GHz')
            ar=popt[1] #popt[2]*popt[1] <- gaussian case
            δr=perr[1] #perr[2]*popt[1]+popt[2]*perr[1] <- gaussian case
            al=popt[4] #popt[5]*popt[4] <- gaussian case
            δl=perr[4] #perr[5]*popt[4]+popt[5]*perr[4] <- gaussian case
            print('Area=',ar,'±',δr,' and ',al,'±',δl) #Area=a*σ=  <- gaussian case
            #print('The ratio between the peak\'s area is (left/right):',al/ar,'±',(δr*al+δl*ar)/ar**2)
    else: #without nearby Carbon-13
        if nG==1:
            print('New MW Freq should be: ',(mwFreq*1e3 - nuphi + popt[2])/1e3,'GHz')
        elif nG==2:
            print('Δν mI±1,0:',-popt[2]+popt[5],'MHz')
            print('Area of mI=+1/All areas = ',(popt[4])/(popt[1]+popt[4]),'±',(popt[4]*(perr[1]+perr[4]))/(popt[1]+popt[4])**2 +  (perr[4])/(popt[1]+popt[4]) )
            print('New MW Freq should be: ',(mwFreq*1e3 - nuphi + popt[[2,5]].mean())/1e3,'GHz (obtained as mean of 2 peaks)')
            print('New MW Freq should be: ',(mwFreq*1e3 - nuphi + popt[5])/1e3,'GHz (last peak)')
            print('New MW Freq should be: ',(mwFreq*1e3 - nuphi + popt[5] + (popt[2]-popt[5])*(1-popt[4]/(popt[1]+popt[4])) )/1e3,'GHz (obtained as mean of 2 peaks weighted with the areas ratio)')
        elif nG==3:
            print('Δν mI±1,0:',-popt[2]+popt[5],-popt[5]+popt[8],'MHz')
            print('Area of mI=+1/All areas = ',(popt[7])/(popt[1]+popt[4]+popt[7]),'±',(popt[7]*(perr[1]+perr[4]+perr[7]))/(popt[1]+popt[4]+popt[7])**2 +  (perr[7])/(popt[1]+popt[4]+popt[7]) )
            print('New MW Freq should be: ',(mwFreq*1e3 - nuphi + popt[[2,5,8]].mean())/1e3,'GHz (obtained as mean of 3 peaks)')
    plt.show()
    if retRes:
        return popt, perr, r2, optVar
    else:
        return popt, perr, r2



def ramsey_auto(fileName, dataType, mwFreq1, nG=1, splittings=[], nuphi=7.5, plot=False, return_FFT=False, return_DATA=False,retRes=False,add_zeros=True,twoSG=False, mwFreq2=0,skRows=globalskRows):
    """
    Evaluation function for a Ramsey measurement
    You can choose how many frequencies should be fitted and give their initial guess for the splitting. All other
    parameters are calculated automatically.
    """

    print(fileName[-45:])
    if ((dataType!=0)&(dataType!=1))&(dataType!=2):
        return 'Error: dataType should be 0 or 1 or 2 (0=signal  1=signal+ref+ref  2=signal+ref)'
    elif len(splittings) < nG - 1:
        return 'Error: Only ' + str(len(splittings)) + ' splittings given for fitting ' + str(nG) + ' peaks'

    dataAll=openDatFile(fileName,skRows=skRows)
    if not(twoSG):
        ##reRead0:
        if dataType==0:
            col1=dataAll.shape[1]-2
            if plot:
                print('Signal mean value =', dataAll[:,col1].mean())
                plt.errorbar(dataAll[:,0],dataAll[:,col1],yerr=dataAll[:,col1+1],fmt='o',ls='--',label='signal')
                plt.show()
            data1 = np.array([dataAll[:,0]*1e6,dataAll[:,col1],dataAll[:,col1+1]])
            #data1 = reRead0(dataAll,True)
        ##reRead2:
        elif dataType==1:
            col1=dataAll.shape[1]-6
            if plot:
                print('Signal mean value =', dataAll[:,col1+2].mean())
                plt.errorbar(dataAll[:,0],dataAll[:,col1+4],yerr=dataAll[:,col1+5],fmt='ko',ls='-',label='ms=0')
                plt.errorbar(dataAll[:,0],dataAll[:,col1+2],yerr=dataAll[:,col1+3],fmt='o',ls='--',label='signal')
                plt.errorbar(dataAll[:,0],dataAll[:,col1],yerr=dataAll[:,col1+1],fmt='ko',ls='-',label='ms=-1')
                plt.show()
            data1 = np.array([dataAll[:,0]*1e6,dataAll[:,col1+2],dataAll[:,col1+3]])
        elif dataType==2:
            col1=dataAll.shape[1]-4
            if plot:
                print('Signal mean value =', dataAll[:,col1+2].mean())
                plt.errorbar(dataAll[:,0],dataAll[:,col1+2],yerr=dataAll[:,col1+3],fmt='o',ls='-',label='signal')
                plt.errorbar(dataAll[:,0],dataAll[:,col1],yerr=dataAll[:,col1+1],fmt='o',ls='-',label='ms=0')
                plt.show()
            data1 = np.array([dataAll[:,0]*1e6,dataAll[:,col1+2]/dataAll[:,col1],0*dataAll[:,col1+3]])
        #
        if return_DATA:
            return data1
    else:
        ##reRead0:
        if dataType==0:
            col1=dataAll.shape[1]-4
            if plot:
                print('Signal mean value =', dataAll[:,col1].mean())
                plt.errorbar(dataAll[:,0],dataAll[:,col1],yerr=dataAll[:,col1+1],fmt='o',ls='--',label='signal')
                plt.errorbar(dataAll[:,0],dataAll[:,col1+2],yerr=dataAll[:,col1+3],fmt='o',ls='--',label='signal')
                plt.show()
            data1 = np.array([dataAll[:,0]*1e6,dataAll[:,col1],dataAll[:,col1+1]])
            data2 = np.array([dataAll[:,0]*1e6,dataAll[:,col1+2],dataAll[:,col1+3]])
            #data = reRead0(dataAll,True)
        ##reRead2:
        elif dataType==1:
            col1=dataAll.shape[1]-8
            if plot:
                print('Signal mean value =', dataAll[:,col1+2].mean())
                plt.errorbar(dataAll[:,0],dataAll[:,col1+6],yerr=dataAll[:,col1+7],fmt='ko',ls='-',label='ms=0')
                plt.errorbar(dataAll[:,0],dataAll[:,col1+4],yerr=dataAll[:,col1+5],fmt='o',ls='--',label='SG2')
                plt.errorbar(dataAll[:,0],dataAll[:,col1+2],yerr=dataAll[:,col1+3],fmt='o',ls='--',label='SG1')
                plt.errorbar(dataAll[:,0],dataAll[:,col1],yerr=dataAll[:,col1+1],fmt='ko',ls='-',label='ms=-1')
                plt.show()
            data1 = np.array([dataAll[:,0]*1e6,dataAll[:,col1+2],dataAll[:,col1+3]])
            data2 = np.array([dataAll[:,0]*1e6,dataAll[:,col1+4],dataAll[:,col1+5]])
        if return_DATA:
            return data1,data2

    freq1, ampFT1, sine_estimates = fftaux(data1[0,:], data1[1,:], sP=False, return_estim=True,add0s=add_zeros)
    # estimate initial parameters; p01 = [offset, amplitude, peak freq1uency, width]
    p01 = np.array([np.mean(ampFT1), np.max(ampFT1)-np.mean(ampFT1), sine_estimates[2], 0.08])
    if twoSG:
        freq2, ampFT2, sine_estimates = fftaux(data2[0,:], data2[1,:], sP=False, return_estim=True,add0s=add_zeros)
        # estimate initial parameters; p0 = [offset, amplitude, peak frequency, width]
        p02 = np.array([np.mean(ampFT2), np.max(ampFT2)-np.mean(ampFT2), sine_estimates[2], 0.08])
        if return_FFT:
            return freq1, ampFT1, freq2, ampFT2
    if return_FFT:
        return freq1, ampFT1

    if twoSG:
        signals=2
        data_list=[data1,data2]
        p0_list = [p01,p02]
        freq_list= [freq1,freq2]
        ampFT_list=[ampFT1,ampFT2]
        mwFreq_list = [mwFreq1,mwFreq2]
    else:
        signals=1
        data_list=[data1]
        p0_list = [p01]
        freq_list= [freq1]
        ampFT_list=[ampFT1]
        mwFreq_list = [mwFreq1]

    results = []
    for si in range(signals):
        data = data_list[si]
        p0 =  p0_list[si]
        freq=  freq_list[si]
        ampFT= ampFT_list[si]
        mwFreq = mwFreq_list[si]

        plt.figure()
        plt.plot(freq,ampFT,'.:')
        plt.xlim([0, max(freq)])
        plt.xlabel(r'MW freq (MHz)',fontsize=22)
        plt.ylabel(r'Fluorescence intensity (kcps)',fontsize=18)
        plt.title(fileName[-45:])

        if nG==2:
            funcTemp=func2Lorentz
            p0=np.concatenate((p0,[p0[1], p0[2]+splittings[0], p0[3]]))
        elif nG==3:
            funcTemp=func3Lorentz
            p0=np.concatenate((p0,[p0[1], p0[2]+splittings[0], p0[3]],
                [p0[1],p0[2]+splittings[0]+splittings[1], p0[3]]))
        elif nG==4:
            funcTemp=func4Lorentz
            p0=np.concatenate((p0,[p0[1], p0[2]+splittings[0], p0[3]],
                [p0[1], p0[2]+splittings[0]+splittings[1], p0[3]],
                [p0[1], p0[2]+splittings[0]+splittings[1]+splittings[2], p0[3]]))
        elif nG==6:
            funcTemp=func6Lorentz
            p0=np.concatenate((p0,[p0[1], p0[2]+splittings[0], p0[3]],
                [p0[1], p0[2]+splittings[0]+splittings[1], p0[3]],
                [p0[1], p0[2]+splittings[0]+splittings[1]+splittings[2], p0[3]],
                [p0[1], p0[2]+splittings[0]+splittings[1]+splittings[2]+splittings[3], p0[3]],
                [p0[1], p0[2]+splittings[0]+splittings[1]+splittings[2]+splittings[3]+splittings[4], p0[3]]))
        else:
            funcTemp=funcLorentz

        #fitting nG lorentzians:
        popt, perr, r2, *optVar  = fit_func(funcTemp,freq,ampFT,p0,retRes=retRes)
        if add_zeros:
            plt.plot(freq,funcTemp(freq,*popt))
        else:
            xx=np.linspace(0,freq.max(),1000)
            plt.plot(xx,funcTemp(xx,*popt))

        text1='peaks in:'
        text2='MW frequencies:'
        for i in range(nG):
            text1=text1+'\n'+str(popt[3*i+2])+' ± '+str(perr[3*i+2])+' MHz'
            text2=text2+'\n'+str((mwFreq*1e3 - nuphi + popt[3*i+2])/1e3)+' ± '+str((perr[3*i+2])/1e3)+' GHz'
        print(text1)
        print(text2)

        plt.show()

        results.append(popt)
        results.append(perr)
        results.append(r2)
        if retRes:
            results.append(optVar)
    return results



def rabi(file, p0=None, doFit=True, dephase=False, ref=0, nuclearRabi=False, retRes=False,xfact=1e6,yfact=1e-3,skRows=globalskRows):
    # default function to fit a sin² to the rabiNEW10-switchIQ sequence
    data=openDatFile(file,skRows=skRows)
    if nuclearRabi:
        if ref==0:
            col1=data.shape[1]-4
            refdata = None
            refdata_bis = None
        else:
            col1=data.shape[1]-6
            refdata=np.array([data[:,data.shape[1]-2],data[:,data.shape[1]-1]]) * yfact
            refdata_bis = None
        data0 = np.array([data[:,0],data[:,col1],data[:,col1+1]])
        data1 = np.array([data[:,0],data[:,col1+2],data[:,col1+3]])
        data2 = (data0+data1)/2
        data2[2,:]=data2[2,:]/np.sqrt(2)
    else:
        if ref==0:
            col1=data.shape[1]-2
            refdata = None
            refdata_bis = None
        elif ref==1:
            col1=data.shape[1]-4
            refdata=np.array([data[:,data.shape[1]-2],data[:,data.shape[1]-1]]) * yfact
            refdata_bis = None
        else:
            col1=data.shape[1]-6
            refdata    =np.array([data[:,data.shape[1]-4],data[:,data.shape[1]-3]]) * yfact
            refdata_bis=np.array([data[:,data.shape[1]-2],data[:,data.shape[1]-1]]) * yfact
        data2=np.array([data[:,0],data[:,col1],data[:,col1+1]])

    if not(doFit):
        plt.errorbar(data2[0,:]*xfact,data2[1,:]*yfact,yerr=data2[2,:]*yfact,fmt='o',ls=':')
        if ref==1:
            plt.errorbar(data2[0,:]*xfact,refdata[0,:]*yfact,yerr=refdata[1,:]*yfact,fmt='o',ls=':')
        elif ref==2:
            plt.errorbar(data2[0,:]*xfact,refdata[0,:]*yfact,yerr=refdata[1,:]*yfact,fmt='o',ls=':')
            plt.errorbar(data2[0,:]*xfact,refdata_bis[0,:]*yfact,yerr=refdata_bis[1,:]*yfact,fmt='o',ls=':')
        plt.xlabel(r'MW time ($\mu$s)',fontsize=22)
        plt.ylabel(r'Fluorescence intensity (kcps)',fontsize=18)
        plt.show()
    else:
        xdata, ydata, yderr = data2[0, :] * xfact, data2[1, :] * yfact, data2[2, :] * yfact
        return rabiAnalysis(xdata, ydata, yderr=yderr, refdata0=refdata, refdata1=refdata_bis, p0=p0, dephase=dephase, file=file, retRes=retRes)


def rabiAnalysis(xdata, ydata, yderr=None, refdata0=None, refdata1=None, p0=None, dephase=False, file='', retRes=False, xlab=r'MW time ($\mu$s)',ylab=r'Fluorescence intensity (kcps)'):
    if p0 is None:
        xfft, yfft, fft_estims = fftaux(xdata, ydata, sP=False, return_estim=True)
        p0 = np.array([np.mean(ydata), ydata.max()-np.mean(ydata), 1/fft_estims[2]/2, fft_estims[3]])
        print(p0)
    def nutn(x,y0,a,t,ph):  #function to fit
        return y0+a*np.cos(np.pi*x/t-ph)
    popt, perr, r2, *optVar = fit_func(nutn, xdata, ydata, p0, yderr=yderr, retRes=retRes)
    # p0,popt = [offset, amplitude, pi-pulse length, phase]

    #For the first minimum:
    tt=popt[2]
    pha=popt[3]
    if dephase:
        xmin=tt*(pha/np.pi) # because of the phase, it sometimes is without the one
        xzero=tt*(-1/2+pha/np.pi) # because of the phase, it sometimes is with a possitive 1/2
    else:
        xmin=tt*(1+pha/np.pi)
        xzero=tt*(1/2+pha/np.pi)
    plt.figure()#figsize=[8,6])
    plt.errorbar(xdata,ydata,yerr=yderr,fmt='o',ls='',color='grey')
    if refdata0 is not None:
        plt.errorbar(xdata,refdata0[0,:],yerr=refdata0[1,:],fmt='o',ls=':')
    if refdata1 is not None:
        plt.errorbar(xdata,refdata1[0,:],yerr=refdata1[1,:],fmt='o',ls=':')
    xx=np.linspace(xdata[0],xdata[-1],100);
    plt.plot(xx,nutn(xx,*popt),lw=2)
    plt.plot([xmin,xmin],[0.95*ydata.min(),1.05*ydata.max()],ls='--',color='red')
    plt.plot([xzero,xzero],[0.95*ydata.min(),1.05*ydata.max()],ls='--',color='red')
    plt.plot([xdata[0],xdata[-1]],[popt[0],popt[0]],ls='-',color='black')
    plt.grid()
    plt.xlabel(xlab,fontsize=22)
    plt.ylabel(ylab,fontsize=18)
    plt.title(file[-46:])
    plt.show()
    print('1st minimum at x = %2.5f'% (xmin),'us')
    print('1st zero at    x = %2.5f'% (xzero),'us')
    print('{:<7s}{:>8s}{:>12s}{:>11s}{:>8s}'.format('', 'offset', 'amplitude', 'pi-pulse', 'phase'))
    print('{:<7s}{:>8.4f}{:>12.3f}{:>11.5f}{:>8.3f}'.format('value',popt[0],popt[1],popt[2],popt[3]))
    print('{:<7s}{:>8.4f}{:>12.3f}{:>11.5f}{:>8.3f}'.format('error',perr[0],perr[1],perr[2],perr[3]))
    print('R² = %.4f'% (r2), '\nRabi freq = %2.4f'% (1/(2*popt[2])),'MHz',
          '\ncontrast =', '%4.2f'% (2*popt[1]/(popt[0]+popt[1])*100), '%')
    if retRes:
        return popt, perr, r2, optVar
    else:
        return popt, perr, r2



# from https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-or-equal-to-n-in-python
def shift_bit_length(x):
    """Function to find the smallest power of 2 greater than or equal to n"""
    return 1<<(x-1).bit_length()

def fftaux(tdat,ydat,sP=True,add0s=True,return_estim=False):
    dd=ydat-ydat.mean() #remove background
    dd=np.concatenate(( dd,np.zeros(shift_bit_length(len(dd)) - len(dd)) )) # add zeros to get len(data) = a power of 2 (good for FFT)
    #if add0s:#add zeros to increase resolution
    for i in range(int(add0s)):#add zeros to increase resolution
        #dd=np.concatenate((dd,np.zeros(len(dd)),np.zeros(len(dd)),np.zeros(len(dd)),np.zeros(len(dd))))
        dd=np.concatenate(( dd,np.zeros(shift_bit_length(len(dd)+1) - len(dd)) )) # add zeros to get to the next power of 2 (again)
    dt=(tdat[1:]-tdat[:-1]).mean() #time step in μs
    N=len(dd) #number of values

    ft=np.fft.fft(dd)*dt #calculate the fft
    freq = np.fft.fftfreq(N, dt) # generate freq array
    freq = freq[:int(N/2+1)] #take only half of the freq array (the positive part)

    ampFT=np.abs(ft[:int(N/2+1)]) #amplitude of the fft
    if sP:
        plt.figure()
        plt.plot(freq,ampFT,'.',ls='-')
        plt.xlim([0, max(freq)])
        plt.xlabel(r'Freq (MHz)',fontsize=22)
        plt.ylabel(r'FFT',fontsize=18)
        plt.show()
    if return_estim:
        peak_index = ampFT.argmax()
        phase = np.angle(ft[peak_index])
        sig_freq = freq[peak_index]
        # estimation values c
        return freq, ampFT, [ydat.mean(),(ydat.max()-ydat.min())/2,sig_freq ,phase]
    else:
        return freq,ampFT



def funcGauss(x,y0,a,xc,w):
    """return y0 + a*np.exp(-0.5*((x-xc)/w)**2)"""
    return y0 + a*np.exp(-0.5*((x-xc)/w)**2)
def func2Gauss(x,y0,a,xc,w,a1,xc1,w1):
    return y0 + funcGauss(x,0,a,xc,w) + funcGauss(x,0,a1,xc1,w1)
def func3Gauss(x,y0,a,xc,w,a1,xc1,w1,a2,xc2,w2):
    return y0 + funcGauss(x,0,a,xc,w) + funcGauss(x,0,a1,xc1,w1) + funcGauss(x,0,a2,xc2,w2)
def funcLorentz(x,y0,a,xc,w):
    """return y0 + a*(2/np.pi)*w/(4*(x-xc)**2+w**2)"""
    return y0 + a*(2/np.pi)*w/(4*(x-xc)**2+w**2)
def func2Lorentz(x,y0,a,xc,w,a1,xc1,w1):
    return y0 + funcLorentz(x,0,a,xc,w) + funcLorentz(x,0,a1,xc1,w1)
def func3Lorentz(x,y0,a,xc,w,a1,xc1,w1,a2,xc2,w2):
    return y0 + funcLorentz(x,0,a,xc,w) + funcLorentz(x,0,a1,xc1,w1) + funcLorentz(x,0,a2,xc2,w2)
def func4Lorentz(x,y0,a,xc,w, a1,xc1,w1, a2,xc2,w2, a3,xc3,w3,):
    return y0 + funcLorentz(x,0,a,xc,w) + funcLorentz(x,0,a1,xc1,w1) + funcLorentz(x,0,a2,xc2,w2) + funcLorentz(x,0,a3,xc3,w3)
def func6Lorentz(x,y0,a,xc,w, a1,xc1,w1, a2,xc2,w2, a3,xc3,w3, a4,xc4,w4, a5,xc5,w5):
    return y0 + funcLorentz(x,0,a,xc,w) + funcLorentz(x,0,a1,xc1,w1) + funcLorentz(x,0,a2,xc2,w2) + funcLorentz(x,0,a3,xc3,w3) + funcLorentz(x,0,a4,xc4,w4) + funcLorentz(x,0,a5,xc5,w5)



def weightAvg(x,s):
    mu = (x/s**2).sum()/(1/s**2).sum()
    # The error it will return is sum in quadrature of  σ_w & σ_{\bar{x}}
    # σ_w  (weighted sample variance):
    wsv = np.sqrt( ((x**2/s**2).sum()/(1/s**2).sum()-mu**2)/(len(x)-1) )
    # σ_{\bar{x}} (standard error of the weighted mean (with variance weights))
    sewm = np.sqrt(1/(1/s**2).sum())
    return mu, np.sqrt(wsv**2+sewm**2)



def estimators_gaussian(xdata,ydata):
    ## Calculate the estimatorsf of a 1D gaussian distribution
    offset = ydata.mean()-ydata.std()
    amplitude = ydata.max()-offset
    center = xdata[ydata.argmax()]
    width = abs(center - xdata[find_nearest(ydata-offset,amplitude/(np.e**2))[1]])
    p0 = [offset,amplitude,center,width]
    return p0

def fit_gaussian(xdata,ydata,yderr=None,p0=None,retRes=False):
    ## fit of a 1D gaussian distribution
    if p0 is None: # with estimators
        p0 = estimators_gaussian(xdata,ydata)
    fitResult = fit_func(funcGauss,xdata,ydata,p0,yderr=yderr,retRes=retRes)
    return fitResult

def funcCos(x,y0,a,nu,ph):
    return y0+a*np.cos(2*np.pi*nu*x-ph)

def fit_cos(xdata,ydata,yderr=None,p0=None,retRes=False):
    ## fit of a 1D cosine
    if p0 is None:  # with estimators
        xfft, yfft, p0 = fftaux(xdata, ydata, sP=False, return_estim=True)
    fitResult = fit_func(funcCos,xdata,ydata,p0,yderr=yderr,retRes=retRes)
    return fitResult

def funcLine(x,m,x0):
    return m*(x-x0)
def fit_line(xdata,ydata,yderr=None,p0=None,retRes=False):
    ## fit of a 1D line
    if p0 is None:  # with estimators
        m = ((ydata[:-1]-ydata[1:])/(xdata[:-1]-xdata[1:])).mean()
        x0 = (xdata - ydata/m).mean()
        p0 = [m,x0]
    fitResult = fit_func(funcLine,xdata,ydata,p0,yderr=yderr,retRes=retRes)
    return fitResult


def func3GaussN14(x,y0,a,xc,w,a1,a2):
    hf_splitting = 2.15e6 # hyperfine splitting for a N14 spin
    return y0 + funcGauss(x,0,a,xc,w) + funcGauss(x,0,a1,xc+hf_splitting,w) + funcGauss(x,0,a2,xc+2*hf_splitting,w)
