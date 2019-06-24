# This code is for creating the function "simJarzy", which is used to simulate the experiments done
# for the QJE project. It can simulate the usual Jarzy or the generalized one.

import numpy as np
from matplotlib.pyplot import subplots,show,legend,plot

from lens.ELvls import eLvls_MW_R_opt,n_R
from lens.ELvls import drMan,drPhe,drRob,drRob2,drTet,drWol,drDqt

#
# Some useful initial states (not used on this script, but still useful)
#
# Eigenstate (|↑>) of the Hamiltonian Ω σx + δ σz
def upS(alpha):
    res = np.zeros([7,7])
    res[0,0] = np.cos((np.pi/2)*alpha)**2
    res[1,1] = np.sin((np.pi/2)*alpha)**2
    res[0,1] = -np.sin((np.pi/2)*alpha)*np.cos((np.pi/2)*alpha)
    res[1,0] = -np.sin((np.pi/2)*alpha)*np.cos((np.pi/2)*alpha)
    return res
# Eigenstate (|↓>) of the Hamiltonian Ω σx + δ σz
def downS(alpha):
    res = np.zeros([7,7])
    res[0,0] = np.sin((np.pi/2)*alpha)**2
    res[1,1] = np.cos((np.pi/2)*alpha)**2
    res[0,1] = np.sin((np.pi/2)*alpha)*np.cos((np.pi/2)*alpha)
    res[1,0] = np.sin((np.pi/2)*alpha)*np.cos((np.pi/2)*alpha)
    return res
# Eigenstate (|+>) of the Observable
def plusS(alpha):
    res = np.zeros([7,7])
    res[0,0] = np.cos((np.pi/2)*(1/2-alpha))**2
    res[1,1] = np.sin((np.pi/2)*(1/2-alpha))**2
    res[0,1] = np.sin((np.pi/2)*(1/2-alpha))*np.cos((np.pi/2)*(1/2-alpha))
    res[1,0] = np.sin((np.pi/2)*(1/2-alpha))*np.cos((np.pi/2)*(1/2-alpha))
    return res
# Eigenstate (|->) of the Observable
def minusS(alpha):
    res = np.zeros([7,7])
    res[0,0] = np.sin((np.pi/2)*(1/2-alpha))**2
    res[1,1] = np.cos((np.pi/2)*(1/2-alpha))**2
    res[0,1] = -np.sin((np.pi/2)*(1/2-alpha))*np.cos((np.pi/2)*(1/2-alpha))
    res[1,0] = -np.sin((np.pi/2)*(1/2-alpha))*np.cos((np.pi/2)*(1/2-alpha))
    return res
# State with population only on the ms=0 ground state
psi0ms0 = np.array([0,1,0,0,0,0,0])
rho0ms0 = np.diag(psi0ms0**2) # initial state can be density matrix or state-vector
# State with population only on the ms=+1 ground state
psi0msp1 = np.array([1,0,0,0,0,0,0])
rho0msp1 = np.diag(psi0msp1**2) # initial state can be density matrix or state-vector
# State with population only on the ms=+1 ground state
psi0msm1 = np.array([0,0,1,0,0,0,0])
rho0msm1 = np.diag(psi0msm1**2) # initial state can be density matrix or state-vector
# Ground state
psi0G = np.array([1,1,1,0,0,0,0])/np.sqrt(3)
rho0G = np.diag(psi0G**2) # initial state can be density matrix or state-vector


## Function that takes an array and puts zeros in the entries with values smaller than 1e-08
def cut(arr):
    mat0s = np.zeros(arr.shape)
    # check for complex values
    if np.allclose(arr.imag,mat0s): # with no imaginary part
        res=np.zeros(arr.shape)
        diffFrom0 = ~np.isclose( arr.real,mat0s )
        res[diffFrom0] = arr[diffFrom0].real
    else: # with imaginary part
        res=np.zeros(arr.shape,dtype=complex)
        diffFrom0real = ~np.isclose( arr.real,mat0s )
        diffFrom0imag = ~np.isclose( arr.imag,mat0s )
        res[diffFrom0real] = arr[diffFrom0real].real
        res[diffFrom0imag] = res[diffFrom0imag] + 1j*arr[diffFrom0imag].imag
    return res

## Function to plot with nice format the populations of the 7 level model
def plotPopul2(tt,populs,retAxes=False,lineW=2):
    if not(np.array_equal(populs.imag,np.zeros(populs.shape))):
        print('Error: ploting populations with imaginary part')
        return -1
    populs=populs.real
    f,(ax1,ax2,ax3) = subplots(1,3,sharey=True,figsize=[12,4])
    #ax1.plot(tt*1e3,laser,color='gray') # laser pulses

    ax1.plot(tt*1e3,populs[:,0],lw=lineW,label='ms=+1',color='b')
    ax1.plot(tt*1e3,populs[:,1],lw=lineW,label='ms=0',color='g')
    ax1.plot(tt*1e3,populs[:,2],lw=lineW,label='ms=-1',color='r')

    ax2.plot(tt*1e3,populs[:,3],lw=lineW,label='ms=+1',color='b')
    ax2.plot(tt*1e3,populs[:,4],lw=lineW,label='ms=0',color='g')
    ax2.plot(tt*1e3,populs[:,5],lw=lineW,label='ms=-1',color='r')

    ax3.plot(tt*1e3,populs[:,6],lw=lineW,label='singlet',color='black')

    ax1.set_ylim(0,1)
    ax1.set_title('Ground')
    ax1.set_xlabel('T [ns]')
    ax1.legend(loc='best')
    ax2.set_title('Excited')
    ax2.set_xlabel('T [ns]')
    ax2.legend(loc='best')
    ax3.set_title('Metastable')
    ax3.set_xlabel('T [ns]')
    ax3.legend(loc='best')
    if retAxes:
        return ax1,ax2,ax3
    else:
        show()

## Function to simulate the experiment (with option to plot all intermediate steps)
def simJarzy(iniSs,tcell,tL,nN,tSt,nu,deltaFact,er,model='Dqt',showAllsteps = True,chBas=[1/4,0],prepSigmax=False):
    # iniSs = List of initial states,
    #         or 'tuple' with length (iniSs[i]*π-pulse) to prepare two initial states (from |0> along σy, and along -σy)
    # tcell = Time between laser pulses
    # tL = duration of the laser pulse
    # nN = Number of laser pulses:
    # tSt = Time step
    # nu = Rabi frequency (on resonance)
    # deltaFact = Detuning of the MW in terms of omeganu (*2*np.pi)
    # er = Excitation rate, in terms of Γ (1 means: excitation_rate = Γ)
    # model (default='Dqt') = Decay rate model
    # showAllsteps (default=True) = Show all the steps in the process?
    # chBas (default=[1/4,0]) = Change basis pulse duration=π*chBas[0] and rotation sign=(-1)**chBas[1] (along σy)
    # prepSigmax (default=False) = prepare and change basis along sigmaX, not along sigmaY (for Observable with delta=0)

    print('Model: ',model)
    if model == 'Rob':
        dr = drRob
    elif model == 'Rob2':
        dr = drRob2
    elif model == 'Tet':
        dr = drTet
    elif model == 'Man':
        dr = drMan
    elif model == 'Phe':
        dr = drPhe
    elif model == 'Wol':
        dr = drWol
    elif model == 'Dqt':
        dr = drDqt
    else:
        dr = model
    w1 = dr[0]/(dr[0]+dr[1])
    w0 = dr[0]/(dr[0]+dr[2])


    # On-resonance Rabi angular frequency
    omega = 2*np.pi*nu
    # Detuning of the MW
    delta = deltaFact*omega #MHz

    times0 = np.arange(0,tcell-tL/2+tSt,tSt) #just before first laser
    times = np.arange(0,tcell-tL+tSt,tSt)
    timesbis = np.arange(0,tL+tSt,tSt)

    print(str(tcell)+' = tcell -- Time between laser pulses in [\mu s]')
    print(str(tL)+' = tL -- Duration of the laser pulse in [\mu s]')
    print(str(nN)+' = nN -- Number of laser pulses')
    print(str(tSt)+' = tSt -- Time-step in [\mu s]')
    print(str(nu)+' = nu -- MW Rabi frequency in [MHz]')
    if omega!=0:
        print(str(delta/omega)+' = delta/omega -- Proportional detuning of the MW')
    print(str(er)+' = er -- Excitation rate, in terms of radiative decay rate')


    prepInitStates = np.isscalar(iniSs[0])
    if prepInitStates:
        # 2 μs of illumination
        mat01  = eLvls_MW_R_opt(rho0G , [0,2.0], er,model, 'rho',0,0,0)
        rhoinit1 = np.reshape(np.array(mat01.expect)[:,-1],(n_R,n_R))

        # 0.75 μs waiting
        mat02  = eLvls_MW_R_opt(rhoinit1.transpose() , [0,0.75], 0,model, 'rho',0,0,0)
        rhoinit2 = np.reshape(np.array(mat02.expect)[:,-1],(n_R,n_R))

        # transforming into the desired initial states
        if not(prepSigmax):
            mat03  = eLvls_MW_R_opt(rhoinit2.transpose() , [0,iniSs[0]/nu/2], 0,model, 'rho',0,-omega,0) # a pulse along -Y
            mat03b = eLvls_MW_R_opt(rhoinit2.transpose() , [0,iniSs[1]/nu/2], 0,model, 'rho',0,omega,0) # a pulse along Y
        else:
            mat03  = eLvls_MW_R_opt(rhoinit2.transpose() , [0,iniSs[0]/nu/2], 0,model, 'rho',-omega,0,0) # a pulse along -X
            mat03b = eLvls_MW_R_opt(rhoinit2.transpose() , [0,iniSs[1]/nu/2], 0,model, 'rho',omega,0,0) # a pulse along X
        rhoinit3  = np.reshape(np.array(mat03.expect)[:,-1],(n_R,n_R))
        rhoinit3b = np.reshape(np.array(mat03b.expect)[:,-1],(n_R,n_R))

        # I remove the values that are different from zero by a factor 1e-08 or smaller:
        rhoinit = cut(rhoinit3)
        rhoinitb= cut(rhoinit3b)

        iniSs = [rhoinit,rhoinitb]

        ##
        ## Calculating the references
        ##
        # a pi pulse to go from ms=0 to ms=+1
        mat02b  = eLvls_MW_R_opt(rhoinit2.transpose() , [0,1/nu/2], 0,model, 'rho',omega,0,0)
        rhoinit2b = np.reshape(np.array(mat02b.expect)[:,-1],(n_R,n_R))
        # Illuminate for 80 ns
        mat0ref  = eLvls_MW_R_opt(rhoinit2.transpose()  ,np.array([0,0.08]), er,model, 'rho',0,0,0)
        mat0refb = eLvls_MW_R_opt(rhoinit2b.transpose() ,np.array([0,0.08]), er,model, 'rho',0,0,0)
        ref0rho  = np.reshape(np.array(mat0ref.expect)[:,-1],(n_R,n_R))
        refp1rho = np.reshape(np.array(mat0refb.expect)[:,-1],(n_R,n_R))
        # PL calculated with populations
        ref0 = (ref0rho[3,3]*w1  +  ref0rho[4,4]*w0  +  ref0rho[5,5]*w1).real
        ref1 = (refp1rho[3,3]*w1  +  refp1rho[4,4]*w0  +  refp1rho[5,5]*w1).real
        # PL calculated with coherences
        refC0 = (ref0rho[3,0]+ref0rho[4,1]+ref0rho[5,2]).imag
        refC1 = (refp1rho[3,0]+refp1rho[4,1]+refp1rho[5,2]).imag



    #rhosFins = list([0,0])
    sigPop = [0]*len(iniSs)
    sigCoh = [0]*len(iniSs)
    for idx,rho0 in enumerate(iniSs): # Initial state
        print('Initial state:\n',rho0)

        tsim=np.array([])
        alldat=np.array([])
        for i in range(nN):
            if i==0:
                ms0rho  = eLvls_MW_R_opt(rho0.transpose() , times0, 0,model, 'rho',omega,0,delta) #times
                alldat = np.array(ms0rho.expect)
                tsim=np.concatenate([tsim, times0])
            else:
                ms0rho  = eLvls_MW_R_opt(rho0fin.transpose(),times, 0,model, 'rho',omega,0,delta)
                alldat = np.concatenate([alldat , np.array(ms0rho.expect)],1)
                tsim=np.concatenate([tsim, times0[-1] + (i-1)*(times[-1]) + i*(timesbis[-1]) + times])
            rho1 = np.reshape(np.array(ms0rho.expect)[:,-1],(n_R,n_R))


            ms0rhobis  = eLvls_MW_R_opt(rho1.transpose() , timesbis, er,model, 'rho',omega,0,delta)
            alldat = np.concatenate([alldat , np.array(ms0rhobis.expect)],1)
            rho1b = np.reshape(np.array(ms0rhobis.expect)[:,-1],(n_R,n_R))
            tsim=np.concatenate([tsim, times0[-1] + i*(times[-1]) + i*(timesbis[-1]) + timesbis])

            rho0fin = rho1b

        ###
        ### This next part is what takes a lot of time
        ###
        if showAllsteps == True:
            popul3  = np.zeros([len(tsim),n_R],dtype=complex)
            popul4 = np.zeros([len(tsim),n_R],dtype=complex)
            popul5 = np.zeros([len(tsim),n_R],dtype=complex)
        rhofinRO  =  np.zeros([len(tsim),n_R*n_R],dtype=complex)
        for i in range(len(tsim)):
            rho2 = np.reshape(alldat[:,i],(n_R,n_R))

            # Waiting 40 ns (decay from excited states)
            rhofin  = eLvls_MW_R_opt(rho2.transpose() ,np.array([0,0.04]), 0,model, 'rho',0,0,0)
            rho3 = np.reshape(np.array(rhofin.expect)[:,-1],(n_R,n_R))

            # Apply change of basis
            if omega!=0:
                if not(prepSigmax): #(π/4 along σY)
                    #rhofin2 = eLvls_MW_R_opt(rho3.transpose() ,np.array([0,1/nu/8]), 0,model, 'rho',0,omega,0)
                    rhofin2 = eLvls_MW_R_opt(rho3.transpose(),np.array([0,chBas[0]/nu/2]),0,model,'rho',0,omega*(-1)**chBas[1],0)
                else: #(π/4 along σX)
                    rhofin2 = eLvls_MW_R_opt(rho3.transpose(),np.array([0,chBas[0]/nu/2]),0,model,'rho',omega*(-1)**chBas[1],0,0)
            else:
                rhofin2 = eLvls_MW_R_opt(rho3.transpose(),np.array([0,0]),0,model,'rho',0,0,0)
            rho4 = np.reshape(np.array(rhofin2.expect)[:,-1],(n_R,n_R))

            # Waiting half μs
            rhofin3 = eLvls_MW_R_opt(rho4.transpose() ,np.array([0,0.5]), 0,model, 'rho',0,0,0)
            rho5 = np.reshape(np.array(rhofin3.expect)[:,-1],(n_R,n_R))

            # Illuminate for 80 ns
            rhofin4 = eLvls_MW_R_opt(rho5.transpose() ,np.array([0,0.08]), er,model, 'rho',0,0,0)
            rhofinRO[i] = np.array(rhofin4.expect)[:,-1]

            if showAllsteps == True:
                popul3[i] = rho3.diagonal()
                popul4[i] = rho4.diagonal()
                popul5[i] = rho5.diagonal()
        #rhosFins[idx] = rhofinRO

        if showAllsteps == True:
            print('The dynamics:')
            plotPopul2(tsim,alldat[np.arange(7)*(n_R+1),:].transpose())
                         #alldatR0[:,2+np.arange(7)*(n_R+1)])
            print('Waiting for each point 40 ns:')
            plotPopul2(tsim,popul3)
            print('Changing basis:')
            plotPopul2(tsim,popul4)
            print('Waiting for each point 500 ns:')
            plotPopul2(tsim,popul5)
            print('Illuminating during 80 ns:')
            plotPopul2(tsim,rhofinRO[:,np.arange(7)*(n_R+1)])


        ##
        ## Collecting info from excited states population
        ##
        sig0aux = rhofinRO[:,(3+np.arange(3))*(n_R+1)].real
        sigPop[idx] = sig0aux[:,0]*w1 + sig0aux[:,1]*w0 + sig0aux[:,2]*w1
        ##
        ## Analyzing also the coherences
        ##
        coh0_30 = abs(rhofinRO[:,np.ravel_multi_index((3,0),(7,7))].imag)
        coh0_41 = abs(rhofinRO[:,np.ravel_multi_index((4,1),(7,7))].imag)
        coh0_52 = abs(rhofinRO[:,np.ravel_multi_index((5,2),(7,7))].imag)
        sigCoh[idx] = coh0_30 + coh0_41 + coh0_52
    #


    if not(prepInitStates):
        ref0 = sigPop[0][0]
        ref1 = sigPop[1][0]

        refC0 = sigCoh[0][0]
        refC1 = sigCoh[1][0]
    if ((-1)**chBas[1])==-1:
        names = ['|->','|+>']
    else:
        names = ['|$\\downarrow$>','|$\\uparrow$>']

    plot(tsim,(sigCoh[0]-refC1)/(refC0-refC1),lw=2,label='coherences '+names[0])
    plot(tsim,(sigCoh[1]-refC1)/(refC0-refC1),lw=2,label='coherences '+names[1])

    plot(tsim,(sigPop[0]-ref1)/(ref0-ref1),lw=2,ls='--',label='weighted populations '+names[0])
    plot(tsim,(sigPop[1]-ref1)/(ref0-ref1),lw=2,ls='--',label='weighted populations '+names[0])

    legend(loc='best') #bbox_to_anchor=(1.4, 1.05))
    show()

    if prepInitStates:
        return tsim,sigPop,sigCoh,[ref0,ref1],[refC0,refC1]
    return tsim,sigPop,sigCoh
