import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from time import strftime

from lens.hyperfine import cpmg
from lens.hyperfine import nested
from lens.hyperfine import nested2
from lens.hyperfine import uhrig
from lens.hyperfine import nestedUhrig

def chiCalc(funcNoise,para,n,rm,timeVec,m=8,simType=cpmg,newOffset=0,sP=False,sP4debug=False,calculate=True,saveSignal=False,notes=''):
    if sP:
        xx=np.linspace(0.1,0.9,1000)*2*np.pi
        plt.figure(figsize=(6,2))
        plt.plot(xx,funcNoise(xx,*para) )
        plt.xlabel(r'$\omega\; [MHz]$',fontsize=18)
        plt.ylabel(r'$S(\omega)\; [MHz]$',fontsize=18)
        plt.show()


    #  Define the spacing between pi-pulses
    distPi = np.ones([m,n])
    extraSE=0
    # fon nested with spin echo we need some extra pulses:
    if simType==nested2:
        distPi = np.ones([m+1,n])
        extraSE=n-1
    if (simType==uhrig)|(simType==nestedUhrig): #uhrig
        if simType==uhrig:
            m=1
        for j in range(1,n+1):
            if m==1:
                distPi[1-1,j-1] = (np.sin(np.pi*j/(2*n+2)))**2
            else:
                for h in range(1,m+1):
                    rh= (np.sin(np.pi/(2*n+2)))**2*(2*h-m-1)/(m-1)
                    distPi[h-1,j-1] = (np.sin(np.pi*j/(2*n+2)))**2 + rm*rh
    elif ((simType==cpmg)|(simType==nested))|(simType==nested2): #equidistant pulses
        for j in range(1,n+1):
            for h in range(1,m+1):
                rh = (2*h-m-1)/(2*m);
                distPi[h-1,j-1] = ((2*j-1)/2+rm*rh)/n
            if (simType==nested2)&(j!=n): #only for nested plus spin echo
                distPi[m,j-1]=j/n
    distPi=distPi.flatten(1) # to turn it into a vector of nxm
    if simType==nested2:
        distPi=distPi[:-1]
    if sP4debug:
        print(distPi*timeVec[0])
        plt.figure(figsize=[10,2])
        plt.plot(distPi,np.ones(len(distPi)),'o',ls='',color='black')
        plt.xlim(0,1)
        plt.xlabel(r'$t/T$',fontsize=14)
        plt.show()

    #Define the filter function:
    def Fn(ω,t):
        return ynRe(ω,t)**2+ynIm(ω,t)**2
    #
    def ynRe(ω,t):
        tempYn = 1 + (-1)**(n*m+1+extraSE)*np.cos(ω*t)
        for j in range(1,n*m+1+extraSE):
            tempYn += 2*(-1)**j*cosn(ω,t,j)
        return tempYn
    #
    def ynIm(ω,t):
        tempYn = (-1)**(n*m+1+extraSE)*np.sin(ω*t)
        for j in range(1,n*m+1+extraSE):
            tempYn += 2*(-1)**j*sinn(ω,t,j)
        return tempYn
    #
    def cosn(ω,t,jj):
        return np.cos(ω*t*distPi[jj-1])
    def sinn(ω,t,jj):
        return np.sin(ω*t*distPi[jj-1])


    #
    if sP4debug: # To see Fn in case something might be wrong
        #t0 = 1; dt = 1; tf = 180;
        #tt=np.arange(t0,tf,dt)
        #plt.plot(tt,Fn(0.5,tt)) #feel free to change ω (first arg of Fn)
        xx=np.arange(0,180,0.2)
        plt.plot(xx,Fn(1,xx))
        #plt.plot(xx,np.sqrt(Fn(xx,30)))
        #plt.ylim(0,50)
        plt.title('Fn',fontsize=16)
        plt.grid()
        plt.show()


    # Define the coherence function
    def Chi(t):
        res, err = np.abs(quad( lambda x: np.abs(funcNoise(x,*para)*Fn(x,t)/(np.pi*x**2)) ,0.001, 8.5)) # integrate from ω=0.001 to ω=8.5
        return res
    #
    if sP4debug:
        ww=np.arange(0.01, 8.5+0.01,0.01)
        tt=2
        plt.plot(ww,np.abs(funcNoise(ww,*para)*Fn(ww,tt)/(np.pi*ww**2)))
        plt.xlabel(r'$\omega/2\pi$ [$MHz$]',fontsize=20)
        plt.title('Integrand of $\chi$',fontsize=18)
        plt.show()
    ######## Here is the end of the definitions ########

    ######## Now we calculate ########
    #timeVec = np.arange(t0,tf,dt)
    signalsimulTable=np.zeros(len(timeVec))

    if newOffset!=0:
        para[0]=newOffset #redefining the offset of the noise spectrum

    if calculate:
        for i in range(len(timeVec)):
            signalsimulTable[i]=Chi(timeVec[i])
            #signalsimulTable[i]=0.5*(1+np.exp(-Chi(timeVec[i])))
            ###signalsimulTable[i]=0.5*(1+np.exp(-(np.pi/2)*Chi(timeVec[i]))) #According to Uhrig
            ###signalsimulTable[i]=0.5*(1+np.exp(-(1/2)*Chi(timeVec[i]))) #According to Zhao
        timestr = strftime("%Y-%m-%d_%H%M%S")

    if saveSignal:
        print(timestr)
        name='./savedData/chiCalc_'+(simType.__name__)+'_N'+str(n)+'_M'+str(m)+'_rm'+str(rm)+'_'+timestr+'.dat'
        print(name)
        np.savetxt(name, [timeVec,signalsimulTable],header=notes)

    return signalsimulTable
