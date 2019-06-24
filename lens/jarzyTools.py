# This code contains auxiliary functions for the analysis of experimental and simulated datasets

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from lens.analysisFunctions import weightAvg
from lens.openDat import openDatFile

##
## To be able to compare with experiments:
##

# for the normalized plots
def normal(data,sp=False,titleText='',refs=True,n2ms1=True):
    nvec=data[:,0]*1e6
    dat1=data[:,[-6,-4,-2]]
    err1=data[:,[-5,-3,-1]]
    if n2ms1: # Usual normalization: ms=+1 --> 1.0 ; ms=0  --> 0.0
        nd = (dat1[:,0]-np.mean(dat1[:,1]))/(np.mean(dat1[:,2])-np.mean(dat1[:,1]))
    else:  # Inverted normalization: ms=0  --> 0.0 ; ms=+1 --> 1.0
        nd = (np.mean(dat1[:,2])-dat1[:,0])/(np.mean(dat1[:,2])-np.mean(dat1[:,1]))
    #ndE= abs((err1[:,0])/(np.mean(dat1[:,1])-np.mean(dat1[:,2])))
    term1=abs(weightAvg(dat1[:,1],err1[:,1])[1]*(dat1[:,0]+np.mean(dat1[:,2])-2*np.mean(dat1[:,1]))/(np.mean(dat1[:,1])-np.mean(dat1[:,2]))**2) #upper limit
    term2=abs(weightAvg(dat1[:,2],err1[:,2])[1]*nd/(np.mean(dat1[:,1])-np.mean(dat1[:,2])))
    term3=abs((err1[:,0])/(np.mean(dat1[:,1])-np.mean(dat1[:,2])))
    ndE= term1+term2+term3
    if sp:
        f,ax = plt.subplots()
        ax.errorbar(nvec,dat1[:,0]*1e-3,yerr=err1[:,0]*1e-3,fmt='.',ls='-') #,color='b'
        ax.errorbar(nvec,dat1[:,1]*1e-3,yerr=err1[:,1]*1e-3,fmt='.',ls='-') #,color='b'
        ax.errorbar(nvec,dat1[:,2]*1e-3,yerr=err1[:,2]*1e-3,fmt='.',ls='-') #,color='b'
        ax.set_xlabel('$T$ $[\mu s]$',fontsize=16)
        ax.set_ylabel('$\mathrm{kcps}$',fontsize=16)
        ax.set_title(titleText,fontsize=18)
        plt.tight_layout()
        plt.show()
    return nvec,nd,ndE

# to normalize and average a set of measurements (in a list)
def normalList(fileList1,normUsual):
    tvecAux,ndAux,ndEAux = np.zeros([3,len(fileList1),len(openDatFile(fileList1[0])[:,0])])
    for i,fi in enumerate(fileList1):
        tvecAux[i],ndAux[i],ndEAux[i] = normal(openDatFile(fi),n2ms1=normUsual[i]) #normalTest(openDatFile(fi))
    return np.array([ tvecAux.mean(0), ndAux.mean(0), np.sqrt(sum(ndEAux**2))/len(ndEAux) ])

lup = r'$\rho_0 =\left|\uparrow\right>\left<\uparrow\right|$'
ldown=r'$\rho_0 =\left|\downarrow\right>\left<\downarrow\right|$'
#lplus = r'$\left|\uparrow\right> + \left|\downarrow\right>$'
#lminus=r'$\left|\uparrow\right> - \left|\downarrow\right>$'
lplus = r'$\rho_0 =\left|+\right>\left<+\right|$'
lminus= r'$\rho_0 =\left|-\right>\left<-\right|$'

##
## To correctly open the ELvls & Matlab simulations:
##
def openSim(f1):
    tsim,downSim,upSim=np.loadtxt(f1,unpack=True)
    with open(f1) as fp:
        refs = [float(num) for num in fp.readline()[8:-3].split(',')]
    return tsim,downSim,upSim,refs

def normSim(tsim,downSim,upSim,refs,n2ms1=True):
    if n2ms1: # Usual normalization: ms=+1 --> 1.0 ; ms=0  --> 0.0
        nddo = (refs[0]-downSim)/(refs[0]-refs[1])
        ndup = (refs[0]-upSim)/(refs[0]-refs[1])
    else:  # Inverted normalization: ms=0  --> 0.0 ; ms=+1 --> 1.0
        nddo = (downSim-refs[1])/(refs[0]-refs[1])
        ndup = (upSim-refs[1])/(refs[0]-refs[1])
    return tsim,nddo,ndup

# Matlab simulations:
def openSimMatBis(f1):
    data = np.loadtxt(f1,delimiter=',')
    if data.shape[1]>3:
        tsim = data[:,0]
        downSim = data[:,np.arange((data.shape[1]-1)/2,dtype=int)*2+1].mean(1)
        upSim = data[:,np.arange((data.shape[1]-1)/2,dtype=int)*2+2].mean(1)
    else:
        tsim,downSim,upSim = np.transpose(data)
    return tsim,downSim,upSim,[1,0]

def openSimMat(f1):
    data = np.loadtxt(f1,delimiter=',')
    if data.shape[1]>3:
        tsim = data[:,0]
        downSim = data[:,np.arange((data.shape[1]-1)/2,dtype=int)*2+1]
        upSim = data[:,np.arange((data.shape[1]-1)/2,dtype=int)*2+2]
    else:
        tsim,downSim,upSim = np.transpose(data)
        print('Warning: just one single simulation')
    return tsim,downSim,upSim,[1,0]

##
## Function to calculate the LHS from the ∣↑⟩ and ∣↓⟩ (∣+⟩ and ∣−⟩)  states
##
def calcLHS(rp,tvec,nd1,nd2,nd1E=[],nd2E=[],factor_bath=0,n2ms1=True,ret_prob_deltas=False):
    #rp=1/(np.e+1) #defined as argument
    rm=1-rp
    factor = -np.log(rm/rp)
    if not(n2ms1):
        nd1=1-nd1
        nd2=1-nd2
    prob_DeltaE_zero = rp*(nd1) + rm*(1-nd2);
    prob_DeltaE_2E = rm*(nd2);
    prob_DeltaE_minus2E = rp*(1-nd1);
    if ret_prob_deltas:
        if len(nd1E)!=len(nd1):
            return prob_DeltaE_zero,prob_DeltaE_2E,prob_DeltaE_minus2E
        return prob_DeltaE_zero,rp*nd1E+rm*nd2E, prob_DeltaE_2E,rm*nd2E, prob_DeltaE_minus2E,rp*nd1E
    lhs = prob_DeltaE_zero + prob_DeltaE_2E*np.exp(factor-factor_bath) + prob_DeltaE_minus2E*np.exp(-factor+factor_bath)
    if len(nd1E)!=len(nd1):
        return lhs
    #errLHS = errP_DeltaE_zero + errP_DeltaE_2E*np.exp(factor) + errP_DeltaE_minus2E*np.exp(-factor);
    #errLHS = abs(rp*(1-np.exp(-factor+factor_bath))*nd1E + rm*(np.exp(factor-factor_bath)-1)*nd2E)
    #   We need to divide by 1/np.sqrt(rp) or 1/np.sqrt(rm)
    #   to correctly take into account the number of times the experiment was performed
    errLHS = abs(rp*(1-np.exp(-factor+factor_bath))*nd1E/np.sqrt(rp)) + abs(rm*(np.exp(factor-factor_bath)-1)*nd2E/np.sqrt(rm))
    return lhs,errLHS

##
## To average the points that have the same number of lasers (even when the time is not the same)
##
# Function to average the points for the same number of applied lasers
def averageLaserPulses(tv1,d1,tau,e1=[],numLas=10):
    if len(e1)==0:
        if len(d1.shape)==1:
            res = np.zeros([numLas,2])
        else:
            res = np.zeros([numLas,2,d1.shape[1]])
        for i in range(numLas):
            idxs = np.arange(len(tv1))[np.array(tv1/tau,dtype=int) == i*np.ones(len(tv1))] # indices for $i number of lasers
            res[i][0] = d1[idxs].mean(0)
            res[i][1] = len(idxs)
    else:
        res = np.zeros([numLas,3])
        for i in range(numLas):
            idxs = np.arange(len(tv1))[np.array(tv1/tau,dtype=int) == i*np.ones(len(tv1))] # indices for $i number of lasers
            res[i][:2] = weightAvg(d1[idxs],e1[idxs])
            res[i][2] = len(idxs)
    return res

##
## Function to calculate the averaged LHS for experimental data and for simulation
##
def calcAvgLHS(rp,tau,fList01,fList02,normUsual=[True,True],factor_bath=0,n2ms1=True, pAxes=[]):
    tvec,ndUp,ndUpE = normalList(fList01,normUsual)
    tvec,ndDo,ndDoE = normalList(fList02,normUsual)
    expLHS=calcLHS(rp,tvec,ndUp,ndDo,ndUpE,ndDoE,factor_bath=factor_bath,n2ms1=n2ms1)
    if len(pAxes)!=0:
        pAxes[0].errorbar(tvec,ndUp,yerr=ndUpE,fmt='.',ls='-')
        pAxes[0].errorbar(tvec,ndDo,yerr=ndDoE,fmt='.',ls='-')
        pAxes[0].set_ylabel(r'$\mathsf{Prob.}\;E_{fin}=E_{\uparrow}$',fontsize=16)
        pAxes[0].set_xlabel('$T$ $[\mu s]$',fontsize=16)
        pAxes[1].errorbar(tvec,expLHS[0],yerr=expLHS[1],fmt='.',ls='-')
        pAxes[1].set_ylabel(r'$\mathsf{LHS}$',fontsize=16)
        pAxes[1].set_xlabel('$T$ $[\mu s]$',fontsize=16)
    return averageLaserPulses(tvec, expLHS[0], tau, e1=expLHS[1])

def calcAvgLHSsim(rp,tau,f01,factor_bath=0,n2ms1=True, pAxes=[],matlabSim=False):
    if matlabSim:
        tsim,doSim,upSim = normSim(*openSimMat(f01),n2ms1=n2ms1)
    else:
        tsim,doSim,upSim = normSim(*openSim(f01),n2ms1=n2ms1)
    simLHS=calcLHS(rp,tsim,upSim,doSim,factor_bath=factor_bath,n2ms1=n2ms1)
    if len(pAxes)!=0:
        pAxes[0].plot(tsim,upSim,'-',lw=2)
        pAxes[0].plot(tsim,doSim,'-',lw=2)
        pAxes[0].set_ylabel(r'$\mathsf{Prob.}\;E_{fin}=E_{\uparrow}$',fontsize=16)
        pAxes[0].set_xlabel('$T$ $[\mu s]$',fontsize=16)
        pAxes[1].plot(tsim,simLHS,'-',lw=2)
        pAxes[1].set_ylabel(r'$\mathsf{LHS}$',fontsize=16)
        pAxes[1].set_xlabel('$T$ $[\mu s]$',fontsize=16)
    return averageLaserPulses(tsim, simLHS, tau)

def calcAvgDE(rp,tau,fList01,fList02,normUsual=[True,True],n2ms1=True):
    tvec,ndUp,ndUpE = normalList(fList01,normUsual)
    tvec,ndDo,ndDoE = normalList(fList02,normUsual)
    pDe0,pDe0err,pDeP2E,pDeP2Eerr,pDeM2E,pDeM2Eerr=calcLHS(rp,tvec,ndUp,ndDo,ndUpE,ndDoE,n2ms1=n2ms1,ret_prob_deltas=True)
    return averageLaserPulses(tvec,pDe0,tau,e1=pDe0err),averageLaserPulses(tvec,pDeP2E,tau,e1=pDeP2Eerr),averageLaserPulses(tvec,pDeM2E,tau,e1=pDeM2Eerr)

def calcAvgDEsim(rp,tau,f01,n2ms1=True,matlabSim=False):
    if matlabSim:
        tsim,doSim,upSim = normSim(*openSimMat(f01),n2ms1=n2ms1)
    else:
        tsim,doSim,upSim = normSim(*openSim(f01),n2ms1=n2ms1)
    pDe0,pDeP2E,pDeM2E = calcLHS(rp,tsim,upSim,doSim,n2ms1=n2ms1,ret_prob_deltas=True)
    return averageLaserPulses(tsim,pDe0,tau),averageLaserPulses(tsim,pDeP2E,tau),averageLaserPulses(tsim,pDeM2E,tau)


##
## Functions to calculate the temperature of the bath
## Which is equivalent to calculate the $p_{th}$ (the initial thermal state) such that ⟨ΔE⟩=0
##
# This function returns:  bath_temp_factor    which is equal to    -2ħω/(k_b T_b)
def factorBath(fList01,fList02, normUsual=[True,True],initX0Min=0.3):
    tvec,ndUp,ndUpE = normalList(fList01,normUsual)
    tvec,ndDo,ndDoE = normalList(fList02,normUsual)
    def funcToMin(ptherm):
        x = ptherm*ndUp + (1-ptherm)*ndDo
        s = ptherm*ndUpE + (1-ptherm)*ndDoE
        mu = (x/s**2).sum()/(1/s**2).sum()
        # returns the weighted sample variance σ_w
        return np.sqrt( ((x**2/s**2).sum()/(1/s**2).sum()-mu**2)/(len(x)-1) )
    rp_th = (minimize(funcToMin,initX0Min).x)[0]
    return -np.log((1-rp_th)/rp_th)

# Same as previous function, but only for simulation.
def factorBathSim(f01, n2ms1=True,matlabSim=False,initX0Min=0.3):
    if matlabSim:
        tsim,doSim,upSim = normSim(*openSimMat(f01),n2ms1=n2ms1)
    else:
        tsim,doSim,upSim = normSim(*openSim(f01),n2ms1=n2ms1)
    if len(doSim.shape)==1:
        def funcToMin(ptherm):
            return (ptherm*upSim + (1-ptherm)*doSim).std() # returns the standard deviation
        rp_th = (minimize(funcToMin,initX0Min).x)[0]
    else:
        rp_th = np.zeros(doSim.shape[1])
        for i in range(len(rp_th)):
            def funcToMin(ptherm):
                return (ptherm*upSim[:,i] + (1-ptherm)*doSim[:,i]).std() # returns the standard deviation
            rp_th[i] = (minimize(funcToMin,initX0Min).x)[0]
    return -np.log((1-rp_th)/rp_th)

# Auxiliary function to calculate temperature in °Kelvin for a selected factor = -2ħω/(k_b T_b)
def calcTemperatureFactor(period,factor,alpha,optstring='',printall=True):
    #NOTE: 'period' has to be the period of the detuned Rabi.
    #factor = -np.log(rm/rp)
    if printall:
        print('period=',period,'μs')
    nu = np.round(np.sin(alpha)/period,3)
    nuErr = np.round((np.sin(alpha)/(period-2e-3) - np.sin(alpha)/(period+2e-3))/2,3)
    if printall:
        print('for Ω/2π=(',nu,'±',nuErr, ')MHz   (assuming ±2ns error in period)')

    delta = np.round(-nu*np.cos(alpha)/np.sin(alpha),3)
    deltaErr = 30e-3
    if printall:
        print('and   δ =(',delta,'±',deltaErr, ')MHz   (assuming ±0.03 MHz error in the detuning)\n')

    omega = (2*np.pi)*np.sqrt(nu**2+delta**2)*1e6
    omegaErr = (2*np.pi)*(abs(delta*deltaErr) + nu*nuErr)*1e6/(2*np.sqrt(nu**2+delta**2))

    kB = 1.38064852e-23
    hbar = 1.054571800e-34

    temperature = -hbar*omega/(kB*factor) # factor = -ħω/kT
    temperatureErr = -hbar*omegaErr/(kB*factor)
    print('T'+optstring+' = -ħω/(k_B factor) = ', np.round(temperature*1e6,1),'±',np.round(abs(temperatureErr)*1e6,1) ,'μK')
    return temperature,temperatureErr

##
##
##
