# This is for functions that you use repeatedly during the analysis of the data
# content with description                                                      -(line):
# movAvgWeight (moving weighted-average)                                        -20
# fit_func (fit a specified function and return fitted parameters,errors & R²)  -56
# normalize (normalize the signal obtainted by the DD sequence)                 -79
# esr (default function to fit a gaussian to the pulsed_esr sequence)           -126
# find_nearest (function to find the nearest value in array (and its position)) -161
# ramsey (function to analyse ramsey measurement:fft and fit lorentzians)       -169
# rabi (function to find the pi pulse fitting a sine^2 to a rabi sequence data) -278
# funcGauss,func2Gauss,func3Gauss (Gaussians)                                   -323:325:327
# funcLorentz,func2Lorentz,func3Lorentz,func4Lorentz,func6Lorentz (Lorentzians) -329:331:333:335:337

from lens.openDat import openDatFile

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def movAvgWeight(ydata,yderr,windowSize):
    # This function calculates the moving average of an array including the errors
    # The function uses the definitions of weighted averages and standard error of the weighted averages given by:
    # (1) Bevington-Robinson_Data reduction and error analysis for the physical sciences   pp 56-58
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




def fit_func(func,xdata,ydata,p0,yderr=[],retChi=False,constr=0):
    if constr==0:
        constr=(-np.inf,np.inf)
    if yderr==[]:
        popt, pcov = curve_fit(func, xdata, ydata, p0,bounds=constr)
    else:
        popt, pcov = curve_fit(func, xdata, ydata, p0,sigma=yderr,bounds=constr)
    #popt, pcov = curve_fit(func, xdata, ydata, [50,230],sigma=yderr)
    perr = np.sqrt(np.diag(pcov))
    #calculation of R²
    residuals = ydata- func(xdata, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata-np.mean(ydata))**2)
    rSquared = 1 - (ss_res / ss_tot)
    if (yderr!=[])&retChi:
        #calculation of χ²  -  see Bevington-Robinson_Data reduction... pp194,195
        xSquared = np.sum(residuals**2/yderr**2)
        return (popt,perr,rSquared,xSquared)
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




def esr(file_path,p0=[40000,-13000,1.1e9,7.5e6],doFit=True):
    # default function to fit a gaussian to the pulsed_esr sequence
    print('\n',file_path[-44:])
    data=openDatFile(file_path)
    col1=data.shape[1]-2
    data2=np.array([data[:,0],data[:,col1],data[:,col1+1]])
    #
    xdata = (data2)[0,:]
    ydata = (data2)[1,:]
    yderr = (data2)[2,:]
    plt.figure(figsize=[8,6])
    plt.errorbar(xdata,ydata,yerr=yderr,fmt='o',ls='',label='ms=-1',color='grey')
    if doFit:
        def funcGauss(x,y0,a,xc,w):
            return y0+a*np.exp(-0.5*((x-xc)/w)**2)
        popt, pcov = curve_fit(funcGauss, xdata, ydata, p0,sigma=yderr)
        perr = np.sqrt(np.diag(pcov))
        residuals = ydata- funcGauss(xdata, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((ydata-np.mean(ydata))**2)
        rSquared = 1 - (ss_res / ss_tot)
        #
        xx=np.linspace(xdata[0],xdata[-1]);
        plt.plot(xx,funcGauss(xx,*popt),lw=2)
    plt.grid()
    plt.xlabel(r'MW freq (GHz)',fontsize=22)
    plt.ylabel(r'Fluorescence intensity (cps)',fontsize=18)
    plt.show()
    if doFit:
        print(['y0','a','xc','w'],'\n', popt ,'\n', perr ,'\nR²=', rSquared,'\n','\n')
        return popt, pcov, rSquared




def find_nearest(array,value):
    #function to find the nearest value in array (and its position):
    idx = (np.abs(array-value)).argmin()
    return array[idx],idx




def ramsey(fileName,dataType, p0,mwFreq,nG=2,nuphi=7.5,plot=False,retData=False,carbon=True):
    print(fileName[-45:])
    if (dataType!=0)&(dataType!=1):
        return 'Error: dataType should be 0 or 1 (0=signal  1=signal+ref+ref)'
    dataAll=openDatFile(fileName)
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
    #
    if ((len(p0)-1)/3)!=nG:
        return 'ERROR: p0 not compatible with '+str(nG)+' gaussians'

    dd=data[1,:]-data[1,:].mean() #remove background
    #add zeros to increase resolution
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
    popt,perr,r2 = fit_func(funcTemp,freq,ampFT,p0)
    plt.plot(freq,funcTemp(freq,*popt))

    text1='peaks in'
    for i in range(nG):
        text1=text1+'\n'+str(popt[3*i+2])+' ± '+str(perr[3*i+2])+' MHz'
    print(text1)

    if carbon:
        if nG==3:
            print('New MW Freq should be: ???')#,(mwFreq*1000 - nuphi+(popt[2]+(popt[5]+popt[5])/2)/2)/1000,'GHz')
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
        elif nG==3:
            print('Δν mI±1,0:',-popt[2]+popt[5],-popt[5]+popt[8],'MHz')
            print('Area of mI=+1/All areas = ',(popt[7])/(popt[1]+popt[4]+popt[7]),'±',(popt[7]*(perr[1]+perr[4]+perr[7]))/(popt[1]+popt[4]+popt[7])**2 +  (perr[7])/(popt[1]+popt[4]+popt[7]) )
            print('New MW Freq should be: ',(mwFreq*1e3 - nuphi + popt[[2,5,8]].mean())/1e3,'GHz (obtained as mean of 3 peaks)')
    plt.show()
    return popt,perr,r2




def rabi(file,p0=[35,6,0.090,-6.35],doFit=True,dephase=False,ref=False):
    # default function to fit a sin² to the rabiNEW10-switchIQ sequence
    data=openDatFile(file)
    if ref:
        col1=data.shape[1]-4
    else:
        col1=data.shape[1]-2
    if not(doFit):
        plt.errorbar(data[:,0]*1e6,data[:,col1]*1e-3,yerr=data[:,col1+1]*1e-3,fmt='o',ls='-')
        plt.xlabel(r'MW time ($\mu$s)',fontsize=22)
        plt.ylabel(r'Fluorescence intensity (kcps)',fontsize=18)
        plt.show()
    else:
        xdata,ydata,yderr = data[:,0]*1e6, data[:,col1]*1e-3, data[:,col1+1]*1e-3
        def nutn(x,y0,a,t,ph): #function to
            return y0+a*np.cos(np.pi*x/t-ph*2*np.pi/360)
        popt, perr, r2 = fit_func(nutn, xdata, ydata, p0,yderr=yderr)

        #For the first minimum:
        tt=popt[2]
        pha=popt[3]
        if dephase:
            xmin=tt*(pha*2/360) # because of the phase, it sometimes is without the one
            xzero=tt*(-1/2+pha*2/360) # because of the phase, it sometimes is with a possitive 1/2
        else:
            xmin=tt*(1+pha*2/360)
            xzero=tt*(1/2+pha*2/360)
        plt.figure(figsize=[8,6])
        plt.errorbar(xdata,ydata,yerr=yderr,fmt='o',ls='',color='grey')
        xx=np.linspace(xdata[0],xdata[-1],100);
        plt.plot(xx,nutn(xx,*popt),lw=2)
        plt.plot([xmin,xmin],[0.95*ydata.min(),1.05*ydata.max()],ls='--',color='red')
        plt.plot([xzero,xzero],[0.95*ydata.min(),1.05*ydata.max()],ls='--',color='red')
        plt.plot([xdata[0],xdata[-1]],[popt[0],popt[0]],ls='-',color='black')
        plt.grid()
        plt.xlabel(r'MW time ($\mu$s)',fontsize=22)
        plt.ylabel(r'Fluorescence intensity (kcps)',fontsize=18)
        plt.show()
        print('1st minimum at x=',xmin,'us')
        print('1st zero at x=',xzero,'us')
        print(['y0','a','t','ph'],'\n', popt ,'\n', perr ,'\nR²=', r2,'\nRabi freq=', 1/(2*popt[2]),'MHz\n','\n')
        return popt, perr, r2



def funcGauss(x,y0,a,xc,w):
    return y0 + a*np.exp(-0.5*((x-xc)/w)**2)
def func2Gauss(x,y0,a,xc,w,a1,xc1,w1):
    return y0 + funcGauss(x,0,a,xc,w) + funcGauss(x,0,a1,xc1,w1)
def func3Gauss(x,y0,a,xc,w,a1,xc1,w1,a2,xc2,w2):
    return y0 + funcGauss(x,0,a,xc,w) + funcGauss(x,0,a1,xc1,w1) + funcGauss(x,0,a2,xc2,w2)
def funcLorentz(x,y0,a,xc,w):
    return y0 + a*(2/np.pi)*w/(4*(x-xc)**2+w**2)
def func2Lorentz(x,y0,a,xc,w,a1,xc1,w1):
    return y0 + funcLorentz(x,0,a,xc,w) + funcLorentz(x,0,a1,xc1,w1)
def func3Lorentz(x,y0,a,xc,w,a1,xc1,w1,a2,xc2,w2):
    return y0 + funcLorentz(x,0,a,xc,w) + funcLorentz(x,0,a1,xc1,w1) + funcLorentz(x,0,a2,xc2,w2)
def func4Lorentz(x,y0,a,xc,w, a1,xc1,w1, a2,xc2,w2, a3,xc3,w3,):
    return y0 + funcLorentz(x,0,a,xc,w) + funcLorentz(x,0,a1,xc1,w1) + funcLorentz(x,0,a2,xc2,w2) + funcLorentz(x,0,a3,xc3,w3)
def func6Lorentz(x,y0,a,xc,w, a1,xc1,w1, a2,xc2,w2, a3,xc3,w3, a4,xc4,w4, a5,xc5,w5):
    return y0 + funcLorentz(x,0,a,xc,w) + funcLorentz(x,0,a1,xc1,w1) + funcLorentz(x,0,a2,xc2,w2) + funcLorentz(x,0,a3,xc3,w3) + funcLorentz(x,0,a4,xc4,w4) + funcLorentz(x,0,a5,xc5,w5)