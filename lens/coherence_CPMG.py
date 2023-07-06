# This is the same as coherence.py, but using the analytical expression for the filter function
# in the particular case of CPMG sequences.
# (see lines 26-35 of this file, and compare with lines 23-77 of coherence.py)

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

def chiCalcCPMG(funcNoise,para,n,timeVec,sP=False,sP4debug=False,calculate=True,saveSignal=False,notes=''):
    if sP:
        xx=np.linspace(0.1,0.9,1000)*2*np.pi
        plt.figure(figsize=(6,2))
        plt.plot(xx,funcNoise(xx,*para) )
        plt.xlabel(r'$\omega\; [MHz]$',fontsize=18)
        plt.ylabel(r'$S(\omega)\; [MHz]$',fontsize=18)
        plt.show()

    #Define the filter function: (see PhysRevB.77.174509, Cywinski (2008))
    #Note: As with respect to Yuge, in this case there is a factor of 2 to make it compatible with our definition
    if n%2==0: # even n
        def Fn(ω,t):
            z=ω*t
            return 2*8*(np.sin(z/(4*n)))**4 * (np.sin(z/2))**2 / (np.cos(z/(2*n)))**2
    else: # odd n
        if n==1:
            def Fn(ω,t):
                z=ω*t
                return 2*8*(np.sin(z/(4)))**4
        else:
            def Fn(ω,t):
                z=ω*t
                return 2*8*(np.sin(z/(4*n)))**4 * (np.cos(z/2))**2 / (np.cos(z/(2*n)))**2

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
        res, err = np.abs(quad( lambda x: np.abs(funcNoise(x,*para)*Fn(x,t)/(np.pi*x**2)) ,0.001, 6.5)) # integrate from ω=0.001 to ω=8.5
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

    if calculate:
        for i in range(len(timeVec)):
            signalsimulTable[i]=Chi(timeVec[i])
            #signalsimulTable[i]=0.5*(1+np.exp(-Chi(timeVec[i])))
            ###signalsimulTable[i]=0.5*(1+np.exp(-(np.pi/2)*Chi(timeVec[i]))) #According to Uhrig
            ###signalsimulTable[i]=0.5*(1+np.exp(-(1/2)*Chi(timeVec[i]))) #According to Zhao
        timestr = strftime("%Y-%m-%d_%H%M%S")

    if saveSignal:
        print(timestr)
        name='./savedData/chiCalc_CPMG_N'+str(n)+timestr+'.dat'
        print(name)
        np.savetxt(name, [timeVec,signalsimulTable],header=notes)

    return signalsimulTable
