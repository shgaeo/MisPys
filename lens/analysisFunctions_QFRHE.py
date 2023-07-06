# Function to analyse and plot the data for quantum fluctuation relations for heat engines (QFRHE)
# Based on function "plotall" from 2019-06-28_TPMP_laser-pulses-and-time-varying-Hamiltonian.ipynb

import numpy as np
import matplotlib.pyplot as plt
#
from lens.analysisFunctions import find_nearest,fit_cos,fit_line

def plotExperSimul_QFRHE(expdata,fmatrix,fsweep,η = 2,
                   calcbetainf=False,deltabeta=True,returnLHS = False,returnRHS = False,tempTau=0,tlim = 3.5,
                   save_plots=False, save_dir=None, fmt='pdf',figureSize = [4.8,3.6],ticksLabelSize = 14,
                   join_Simul=True,eigen_states_labels='arrows'
                  ):
    """
    Function to perform all the analysis of the TPMP experiment:

    Parameters:
    expdata = Experimental data in format: ([tvecUp,ndUp,ndUpE],[tvecDo,ndDo,ndDoE]).
    fmatrix = File-path of matlab simulation (result).
    fsweep = File-path of matlab simulation (sweep).
    ...
    η = Value to put inside the exponential: LHS = P_updown + P_downup + exp(-η)*P_upup + exp(η)*P_downdown. Note that the initial state is recalculated from the asymptotic population in order to make LHS=1
    calcbetainf = If True, calculates asymptotic population.
    deltabeta = If True, normalization factor is \Delta\beta. If False, norm. fact. is \beta.
    returnLHS = If False, function returns nothing
    returnRHS = If True, function returns RHS
    tempTau = If !=0, then plots vertical dashed lines in the times of the laser pulses
    tlim = (in μs) lower time limit to calculate the average of full phase cycles (only used
            if calcbetainf=True, deltabeta=True, and there is phase-sweep)

    Returns: (Only if returnLHS = False)
    ([tvecUp,experlhs,experlhsE],[dynT,lhs,rhs]) #rhs is optional (returnRHS = True)
    """
    # Labels definition:
    eig_sta_lab_1='1'
    eig_sta_lab_2='2'
    if eigen_states_labels=='arrows':
        eig_sta_lab_1=r'\uparrow'
        eig_sta_lab_2=r'\downarrow'
    elif eigen_states_labels=='plusminus':
        eig_sta_lab_1=r'+'
        eig_sta_lab_2=r'-'

    # Experimental data:
    tvecUp,ndUp,ndUpE=expdata[0]
    tvecDo,ndDo,ndDoE=expdata[1]

    # amplitude of the Hamiltonian:
    # H = sweep*(1/2)*omega*(cos(angle_hamiltonian)*sigma_z - sin(angle_hamiltonian)*sigma_x)
    sweep = np.loadtxt(fsweep,delimiter=',')
    # solution of the simulation:
    dynT,avg_popul,avg_popul_2,pΔPP,pΔPM,pΔMP,pΔMM,βInfty_ωF,LHSmatlab = np.loadtxt(fmatrix,delimiter=',').transpose()
    ### changing the simulated probability; we want prob. of ending in |↑>, instead of ending in |↓>
    avg_popul,avg_popul_2 = 1-avg_popul_2,1-avg_popul

    ## Plot of the amplitude and phase sweeps
    plt.figure(figsize=figureSize)
    plt.plot(dynT,sweep)
    plt.legend(['Amplitude sweep', 'Phase sweep'][:len(sweep.shape)],loc='best')
    plt.ylabel('Hamiltonian sweep',fontsize=12)
    plt.xlabel(r'$T$ [$\mu$s]',fontsize=14) #plt.xlabel(r'$n$',fontsize=18)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=ticksLabelSize)
    plt.show()

    ## Calculate the initial and asymptotic beta:
    if len(sweep.shape)==1:
        sweep_amp=sweep
        sweep_phase = None
    else:
        sweep_amp=sweep[:,0]
        sweep_phase=sweep[:,1]
        if np.array_equal(sweep_phase,np.zeros(len(sweep_phase))):
            sweep_phase = None

    a = sweep_amp #sweep_amp[0]+(0-dynT[0])*(sweep_amp[1]-sweep_amp[0])/(dynT[1]-dynT[0]) #2 #1 #
    #pup = pΔPP[0] # = 1/(1+np.e)
    #ω0β0 = np.log((1-pup)/pup) # 2|E_0| x initial beta
    #ωfβ0 = a*ω0β0/a[0] # 2|E_t| x initial beta
    cyclPos = 0 # in this variable we will store the indexes for full cycles of phase sweep
    phPer = 0 # in this variable we will store the phase sweep period
    if deltabeta:
        if calcbetainf: # Calculate the asymptotic beta:
            if sweep_phase is None:
                # p_infty:
                p8 = (avg_popul[-1]+avg_popul_2[-1])/2 #np.mean((avg_popul[(dynT/tau)>26]+avg_popul_2[(dynT/tau)>26])/2) #
                print('P^∞ = ',p8)
            else:
                # find period of the phase sweep:
                if (True in (sweep_phase>(np.pi+1))): #linear sweep
                    line=fit_line(dynT,sweep_phase)[0]
                    phPer = round(line[1] + 2*np.pi/line[0],5)
                else: #cosine sweep
                    phPer = round(1/fit_cos(dynT,sweep_phase)[0][2],5)
                print('calculated phase period:',phPer,'μs')
                #tlim = 3.5 # (in μs) lower time limit to calculate the average of full phase cycles
                # cyclPos is the array of times for full periods to calculate the asymptotic population
                cyclPos = np.array([find_nearest(dynT,tpos)[1] for tpos in np.arange(1,dynT.max()/phPer)*phPer if tpos>tlim],dtype=int)
                # cyclPosAll is the arry of times for full periods to plot together with the data (red crosses)
                cyclPosAll_exper = np.array([find_nearest(dynT,tpos)[1] for tpos in np.arange(1,dynT.max()/phPer)*phPer if tpos<tvecUp.max()],dtype=int)
                cyclPosAll_simul = np.array([find_nearest(dynT,tpos)[1] for tpos in np.arange(1,dynT.max()/phPer)*phPer],dtype=int)
                # p_infty:
                p8 = (avg_popul[cyclPos]+avg_popul_2[cyclPos])/2
                print('P^∞ = ',p8.mean(),'±',p8.std(),'\nAveraged points: 2 x',len(p8))
                p8 = p8.mean()
            ωfβ8 = a*np.log((1-p8)/p8) # 2|E_t| x asymptotic beta
            ω0β8 = a[0]*np.log((1-p8)/p8) # 2|E_0| x asymptotic beta
        else: #case of β_∞=0
            ωfβ8 = 0
            ω0β8 = 0
        #
        # We calculate the initial state from the final state, this way the argument inside the exponential
        # can only take three values 0,+η,-η.
        ω0β0 = η + ω0β8 # 2|E_0| x initial beta
        ωfβ0 = a*ω0β0/a[0] # 2|E_t| x initial beta
        pup = 1/(1+np.exp(ω0β0)) # = 1/(1+np.e)
        print('P0 = ',pup)

    ## Plot of the populations
    plt.figure(figsize=figureSize)
    # plotting the experiment
    lotline,caplines,barlinecols=plt.errorbar(tvecUp,ndUp,yerr=ndUpE,fmt='o',ls='',color='r',label=r'$P_{'+eig_sta_lab_1+r'\!|\,'+eig_sta_lab_1+r'}$',uplims=True,lolims=True,zorder=1) #,alpha=0.5) ,ms=4 #r'init. $\left|E_'+eig_sta_lab_1+r'\right>$'
    caplines[0].set_marker('_')
    caplines[1].set_marker('_')
    lotline,caplines,barlinecols=plt.errorbar(tvecDo,ndDo,yerr=ndDoE,fmt='s',ls='',color='k',label=r'$P_{'+eig_sta_lab_1+r'\!|\,'+eig_sta_lab_2+r'}$',uplims=True,lolims=True,zorder=2) #,alpha=0.5) ,ms=4 r'init. $\left|E_'+eig_sta_lab_2+r'\right>$'
    caplines[0].set_marker('_')
    caplines[1].set_marker('_')
    # plotting the simulation
    if join_Simul:
        prevcolor1 = plt.plot(dynT,avg_popul,lw=2,zorder=3)[0].get_color()
        prevcolor2 = plt.plot(dynT,avg_popul_2,lw=2,color=prevcolor1,zorder=4)[0].get_color()
        if calcbetainf & (not(sweep_phase is None)):
            prevcolor = plt.plot(dynT[cyclPos],avg_popul[cyclPos],'s',label='numerical')[0].get_color()
            plt.plot(dynT[cyclPos],avg_popul_2[cyclPos],'s',color=prevcolor)
            plt.plot([tlim]*2,[0,1],ls='--',lw=1.2,color='grey')
    else:
        prevcolor1 = plt.plot(dynT[cyclPosAll_simul],avg_popul[cyclPosAll_simul],'x',markeredgewidth=2,zorder=3)[0].get_color()
        prevcolor2 = plt.plot(dynT[cyclPosAll_simul],avg_popul_2[cyclPosAll_simul],'x',markeredgewidth=2,zorder=4)[0].get_color()
        if calcbetainf & (not(sweep_phase is None)):
            plt.plot([tlim]*2,[0,1],ls='--',lw=1.2,color='grey')
    plt.ylim(-0.01,1.1)  #(0,1)
    #plt.ylabel(r'$P_{'+eig_sta_lab_1+r'\!|\,i}$',fontsize=16)
    #plt.xlabel('$T$ $[\mu s]$',fontsize=20)
    plt.xlabel(r'$t_\mathrm{f}\;[\mu\mathrm{s}]$',fontsize=16) #plt.xlabel(r'$n$',fontsize=18)
    plt.legend(loc='best',fontsize=16)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=ticksLabelSize)
    if tempTau!=0:
        plt.xlim(0,tvecUp.max())
        ax1=ax.twiny()
        ax1.set_xlim(0,tvecUp.max()/tempTau)
        ax1.set_xlabel(r'$N_L$',fontsize=16)
        lineheight = np.array(ax.get_ylim())*[1.01,0.99]
        ax1.tick_params(axis='both', which='major', labelsize=ticksLabelSize)
        for i in range(int(round(tvecUp.max()/tempTau))):
            ax1.plot([i+1,i+1],lineheight,ls='--',color='gray',alpha=0.5,zorder=0)
        ax1.locator_params(axis='x', nbins=10)
    plt.tight_layout()
    if save_plots:
        plt.savefig(save_dir+'conditional_probs.'+fmt)
    plt.show()

    # Experimental data:
    rp=pup
    rm=1-rp
    pΔPP = rp*avg_popul;
    pΔMM = rm*(1-avg_popul_2);
    pΔMP = rm*avg_popul_2;
    pΔPM = rp*(1-avg_popul);

    expP_DE_PP = rp*ndUp;
    expP_DE_MM = rm*(1-ndDo);
    expP_DE_MP = rm*ndDo;
    expP_DE_PM = rp*(1-ndUp);

    errP_DE_PP = rp*ndUpE;
    errP_DE_MM = rm*ndDoE;
    errP_DE_MP = rm*ndDoE;
    errP_DE_PM = rp*ndUpE;

    ## Plot of the joint probabilities
    plt.figure(figsize=figureSize)

    #prevcolor1 = plt.errorbar(tvecUp,expP_DE_PP,yerr=errP_DE_PP,fmt='o',ms=4,label=r"$P_{'+eig_sta_lab_1+r'\!,\! '+eig_sta_lab_1+r'}$")[0].get_color()
    #prevcolor2 = plt.errorbar(tvecUp,expP_DE_PM,yerr=errP_DE_PM,fmt='o',ms=4,label=r"$P_{'+eig_sta_lab_1+r'\!,\! '+eig_sta_lab_2+r'}$")[0].get_color()
    #prevcolor3 = plt.errorbar(tvecUp,expP_DE_MP,yerr=errP_DE_MP,fmt='o',ms=4,label=r"$P_{'+eig_sta_lab_2+r'\!,\! '+eig_sta_lab_1+r'}$")[0].get_color()
    #prevcolor4 = plt.errorbar(tvecUp,expP_DE_MM,yerr=errP_DE_MM,fmt='o',ms=4,label=r"$P_{'+eig_sta_lab_2+r'\!,\! '+eig_sta_lab_2+r'}$")[0].get_color()
    #if join_Simul:
    #    plt.plot(dynT,pΔPP,lw=2,color=prevcolor1)
    #    plt.plot(dynT,pΔPM,lw=2,color=prevcolor2)
    #    plt.plot(dynT,pΔMP,lw=2,color=prevcolor3)
    #    plt.plot(dynT,pΔMM,lw=2,color=prevcolor4)
    #else:
    #    plt.plot(dynT[cyclPosAll],pΔPP[cyclPosAll],'x',markeredgewidth=1.2,color=prevcolor1)
    #    plt.plot(dynT[cyclPosAll],pΔPM[cyclPosAll],'x',markeredgewidth=1.2,color=prevcolor2)
    #    plt.plot(dynT[cyclPosAll],pΔMP[cyclPosAll],'x',markeredgewidth=1.2,color=prevcolor3)
    #    plt.plot(dynT[cyclPosAll],pΔMM[cyclPosAll],'x',markeredgewidth=1.2,color=prevcolor4)


    if join_Simul:
        prevcolor1 = plt.plot(dynT,pΔPP,lw=2)[0].get_color()
        prevcolor2 = plt.plot(dynT,pΔPM,lw=2)[0].get_color()
        prevcolor3 = plt.plot(dynT,pΔMP,lw=2)[0].get_color()
        prevcolor4 = plt.plot(dynT,pΔMM,lw=2)[0].get_color()
    else:
        prevcolor1 = plt.plot(dynT[cyclPosAll_exper],pΔPP[cyclPosAll_exper],'x',markeredgewidth=2)[0].get_color()
        prevcolor2 = plt.plot(dynT[cyclPosAll_exper],pΔPM[cyclPosAll_exper],'x',markeredgewidth=2)[0].get_color()
        prevcolor3 = plt.plot(dynT[cyclPosAll_exper],pΔMP[cyclPosAll_exper],'x',markeredgewidth=2)[0].get_color()
        prevcolor4 = plt.plot(dynT[cyclPosAll_exper],pΔMM[cyclPosAll_exper],'x',markeredgewidth=2)[0].get_color()
    lotline,caplines,barlinecols=plt.errorbar(tvecUp,expP_DE_PP,yerr=errP_DE_PP,fmt='o',ms=5,label=r'$P_{'+eig_sta_lab_1+r','+eig_sta_lab_1+r'}$',color=prevcolor1,uplims=True,lolims=True)
    caplines[0].set_marker('_')
    caplines[1].set_marker('_')
    lotline,caplines,barlinecols=plt.errorbar(tvecUp,expP_DE_PM,yerr=errP_DE_PM,fmt='o',ms=5,label=r'$P_{'+eig_sta_lab_1+r','+eig_sta_lab_2+r'}$',color=prevcolor2,uplims=True,lolims=True)
    caplines[0].set_marker('_')
    caplines[1].set_marker('_')
    lotline,caplines,barlinecols=plt.errorbar(tvecUp,expP_DE_MP,yerr=errP_DE_MP,fmt='o',ms=5,label=r'$P_{'+eig_sta_lab_2+r','+eig_sta_lab_1+r'}$',color=prevcolor3,uplims=True,lolims=True)
    caplines[0].set_marker('_')
    caplines[1].set_marker('_')
    lotline,caplines,barlinecols=plt.errorbar(tvecUp,expP_DE_MM,yerr=errP_DE_MM,fmt='o',ms=5,label=r'$P_{'+eig_sta_lab_2+r','+eig_sta_lab_2+r'}$',color=prevcolor4,uplims=True,lolims=True)
    caplines[0].set_marker('_')
    caplines[1].set_marker('_')
    plt.ylim(-0.02,1.00)  #(0,1)
    plt.ylabel(r'$P_{j,i}$',fontsize=16)
    #plt.xlabel('$T$ $[\mu s]$',fontsize=20)
    plt.xlabel(r'$t_\mathrm{f}\;[\mu\mathrm{s}]$',fontsize=16) #plt.xlabel(r'$n$',fontsize=18)
    plt.legend(loc='best',fontsize=16,ncol=2)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=ticksLabelSize)
    if tempTau!=0:
        dtx = tvecUp[1]-tvecUp[0]
        plt.xlim(tvecUp.min()-dtx,tvecUp.max()+dtx)
        ax1=ax.twiny()
        ax1.set_xlim((tvecUp.min()-dtx)/tempTau,(tvecUp.max()+dtx)/tempTau)
        ax1.set_xlabel(r'$N_L$',fontsize=16)
        lineheight = np.array(ax.get_ylim())*[1.01,0.99]
        ax1.tick_params(axis='both', which='major', labelsize=ticksLabelSize)
        for i in range(int(round(tvecUp.max()/tempTau))+1):
            ax1.plot([i,i],lineheight,ls='--',lw=1.2,color='gray',alpha=0.5,zorder=0)
        ax1.locator_params(axis='x', nbins=5)
    plt.tight_layout()
    if save_plots:
        plt.savefig(save_dir+'joint_probs.'+fmt)
    plt.show()


    ## Plot of the ΔE probabilities
    plt.figure(figsize=figureSize)

    if join_Simul:
        prevcolor1 = plt.plot(dynT,pΔPP+pΔMM,lw=2)[0].get_color()
        prevcolor2 = plt.plot(dynT,pΔPM,lw=2)[0].get_color()
        prevcolor3 = plt.plot(dynT,pΔMP,lw=2)[0].get_color()
    else:
        prevcolor1 = plt.plot(dynT[cyclPosAll_exper],pΔPP[cyclPosAll_exper]+pΔMM[cyclPosAll_exper],'x',markeredgewidth=2)[0].get_color()
        prevcolor2 = plt.plot(dynT[cyclPosAll_exper],pΔPM[cyclPosAll_exper],'x',markeredgewidth=2)[0].get_color()
        prevcolor3 = plt.plot(dynT[cyclPosAll_exper],pΔMP[cyclPosAll_exper],'x',markeredgewidth=2)[0].get_color()
    lotline,caplines,barlinecols=plt.errorbar(tvecUp,expP_DE_PP+expP_DE_MM,yerr=np.sqrt(errP_DE_PP**2+errP_DE_MM**2)/np.sqrt(2),fmt='o',ms=5,label=r'$\Delta E = 0$',color=prevcolor1,uplims=True,lolims=True)
    caplines[0].set_marker('_')
    caplines[1].set_marker('_')
    lotline,caplines,barlinecols=plt.errorbar(tvecUp,expP_DE_PM,yerr=errP_DE_PM,fmt='o',ms=5,label=r'$\Delta E = +2E_{\theta}$',color=prevcolor2,uplims=True,lolims=True)
    caplines[0].set_marker('_')
    caplines[1].set_marker('_')
    lotline,caplines,barlinecols=plt.errorbar(tvecUp,expP_DE_MP,yerr=errP_DE_MP,fmt='o',ms=5,label=r'$\Delta E = -2E_{\theta}$',color=prevcolor3,uplims=True,lolims=True)
    caplines[0].set_marker('_')
    caplines[1].set_marker('_')
    plt.ylim(-0.02,1.05)  #(0,1)
    plt.ylabel(r'$p(\Delta E)$',fontsize=16)
    #plt.xlabel('$T$ $[\mu s]$',fontsize=20)
    plt.xlabel(r'$t_\mathrm{f}\;[\mu\mathrm{s}]$',fontsize=16) #plt.xlabel(r'$n$',fontsize=18)
    plt.legend(loc=0,fontsize=14)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=ticksLabelSize)
    if tempTau!=0:
        dtx = tvecUp[1]-tvecUp[0]
        plt.xlim(tvecUp.min()-dtx,tvecUp.max()+dtx)
        ax1=ax.twiny()
        ax1.set_xlim((tvecUp.min()-dtx)/tempTau,(tvecUp.max()+dtx)/tempTau)
        ax1.set_xlabel(r'$N_L$',fontsize=16)
        lineheight = np.array(ax.get_ylim())*[1.01,0.99]
        ax1.tick_params(axis='both', which='major', labelsize=ticksLabelSize)
        for i in range(int(round(tvecUp.max()/tempTau))+1):
            ax1.plot([i,i],lineheight,ls='--',lw=1.2,color='gray',alpha=0.5,zorder=0)
        ax1.locator_params(axis='x', nbins=5)
    plt.tight_layout()
    if save_plots:
        plt.savefig(save_dir+'DeltaE_probs.'+fmt)
    plt.show()



    plt.figure(figsize=figureSize)
    if deltabeta:
        # Calculation of the LHS and RHS
        rhs = (np.cosh(ωfβ0/2)/np.cosh(ω0β0/2)) * (np.cosh(ω0β8/2)/np.cosh(ωfβ8/2))
        ## Plot of the LHS & RHS
        factPP = np.exp(-0.5*( ωfβ0-ω0β0-ωfβ8+ω0β8))
        factMM = np.exp(-0.5*(-ωfβ0+ω0β0+ωfβ8-ω0β8))
        factPM = np.exp(-0.5*(-ωfβ0-ω0β0+ωfβ8+ω0β8))
        factMP = np.exp(-0.5*( ωfβ0+ω0β0-ωfβ8-ω0β8))
        #print(0.5*(ωfβ0-ω0β0-ωfβ8+ω0β8),0.5*(-ωfβ0+ω0β0+ωfβ8-ω0β8),0.5*(-ωfβ0-ω0β0+ωfβ8+ω0β8), 0.5*(ωfβ0+ω0β0-ωfβ8-ω0β8))
        lhs = pΔPP*factPP + pΔMM*factMM + pΔPM*factPM + pΔMP*factMP
        plt.plot(dynT,lhs,lw=2,label=r'simulation')
        if calcbetainf:
            plt.plot(dynT,rhs,lw=2,label=r'$e^{-\Delta\beta \Delta F}$')
            plt.ylabel(r'$\langle e^{-\Delta\beta \Delta E} \rangle$',fontsize=18)
            #plt.plot(dynT,2-LHSmatlab,'--',lw=1.2) # I invert because Up & Down definitions are inverted in matlab script
            ## Only if we calculate the asymptotic beta, then the LHS calculated in the matlab code "LHSmatlab"
            ## coincides with the one calculated here "lhs".
            if not(sweep_phase is None):
                #cyclPos2 = np.array([find_nearest(dynT,tpos)[1] for tpos in np.arange(1,dynT.max()/phPer)*phPer],dtype=int)
                #plt.plot(dynT[cyclPos2],lhs[cyclPos2],'o',color='grey',label='LHS $(\Delta F=0)$')
                plt.plot([tlim]*2,[lhs.min(),lhs.max()],ls='--',lw=1.2,color='grey')
        else:
            plt.plot(dynT,rhs,lw=2,label=r'$e^{-\beta_{\mathrm{in}} \Delta F}$')
            plt.ylabel(r'$\langle e^{-\beta_{\mathrm{in}} \Delta E} \rangle$',fontsize=18)

        # Experimental data:
        maskArray = np.array([find_nearest(dynT,tExper)[1] for tExper in tvecUp])
        experlhs = expP_DE_PP*factPP[maskArray] + expP_DE_MM*factMM[maskArray]  +  expP_DE_PM*factPM[maskArray]  +  expP_DE_MP*factMP[maskArray]
        #experlhsE= errP_DE_PP*factPP[maskArray] + errP_DE_MM*factMM[maskArray]  +  errP_DE_PM*factPM[maskArray]  +  errP_DE_MP*factMP[maskArray]
        experlhsE= abs(rp*ndUpE*(factPP-factPM)[maskArray] + rm*ndDoE*(-factMM+factMP)[maskArray])
        plt.errorbar(tvecUp,experlhs,yerr=experlhsE,fmt='o',ms=4,label=r'data')

    else:
        ## Plot of the LHS & RHS
        lhs = pΔPP+pΔMM  +  pΔPM*np.exp(ω0β0)  +  pΔMP*np.exp(-ω0β0)
        rhs = (np.cosh(ωfβ0/2)/np.cosh(ω0β0/2))
        plt.plot(dynT,lhs,lw=2,label=r'simulation')
        if returnRHS:
            plt.plot(dynT,rhs,lw=2,label=r'$e^{-\beta_{\mathrm{in}} \Delta F}$')

        # Experimental data:
        experlhs = expP_DE_PP + expP_DE_MM  +  expP_DE_PM*np.exp(ω0β0)  +  expP_DE_MP*np.exp(-ω0β0)
        #experlhsE= errP_DE_PP + errP_DE_MM  +  errP_DE_PM*np.exp(ω0β0)  +  errP_DE_MP*np.exp(-ω0β0)
        experlhsE= abs(rp*ndUpE*(1-np.exp(ω0β0)) + rm*ndDoE*(-1+np.exp(-ω0β0)))
        plt.errorbar(tvecUp,experlhs,yerr=experlhsE,fmt='o',ms=4,label='LHS')
        plt.ylabel(r'$\langle e^{-\beta_{\mathrm{in}} \Delta E} \rangle$',fontsize=18)



    #plt.xlabel('$T$ $[\mu s]$',fontsize=20)
    plt.xlabel(r'$T$ [$\mu$s]',fontsize=14) #plt.xlabel(r'$n$',fontsize=18)
    plt.legend(loc='best',fontsize=12,ncol=2)
    ax=plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=ticksLabelSize)
    if tempTau!=0:
        plt.xlim(0,dynT.max())
        ax1=ax.twiny()
        ax1.set_xlim(0,dynT.max()/tempTau)
        ax1.set_xlabel(r'$N_L$',fontsize=16)
        lineheight = np.array(ax.get_ylim())*[1.01,0.99]
        ax1.tick_params(axis='both', which='major', labelsize=ticksLabelSize)
        for i in range(int(round(dynT.max()/tempTau))):
            ax1.plot([i+1,i+1],lineheight,ls='--',lw=1.2,color='gray',alpha=0.5)
    plt.tight_layout()
    if save_plots:
        plt.savefig(save_dir+'fluct_relation.'+fmt)
    plt.show()
    if returnLHS:
        if returnRHS:
            return ([tvecUp,experlhs,experlhsE],[dynT,lhs,rhs])
        else:
            return ([tvecUp,experlhs,experlhsE],[dynT,lhs])
