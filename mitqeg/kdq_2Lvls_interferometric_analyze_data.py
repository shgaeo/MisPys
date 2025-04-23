# Script to analyze the data for the interferometric scheme

import numpy as np
from mitqeg.kdq_2Lvls_interferometric import *
from mitqeg.qeg_aux_functions import signal_normalized

# This code is based on the notebook:
# /home/santiago/Documents/Postdoc/2022_Postdoc_MIT/2023_iPython/2023_07/2023_07_25_N_ramsey_Characteristic_function_Hyperfine_analyze_all.ipynb

###########################################
## Analyze experimental data
##
def func_full_analysis(file_list, y0=0,a0=1,ignore_y0a0=False,flip_signal = [],zero_initial_time_u = False,skRows=0,
                       ws=6,sop=False,xText='free evolution $[\mu s]$',n_refs=2,n_signals=4,return_norm_data=2,add_minus_im=False,reduced_contrast=0.1,cutoff=None,use_IFFT=True):
    """
    sop = show original plots
    ws = windowSize for the moving average
    return_norm_data :  0=return normalized data, 1=return n. data with u and (-u) parts, other=return ifft of simulation
    """
    ##########################################################################################
    ##
    ## Open files and normalize data
    ##
    ndata_0_list  =[]
    ndata_1_list  =[]
    errdata_0_list=[]
    errdata_1_list=[]
    #n_movavg_list =[]
    nv_short_0_list = []
    nv_short_1_list = []
    nv_shortE_0_list= []
    nv_shortE_1_list= []

    for i in range(len(file_list)):
        file_path = file_list[i]
        xdat0,ndat,errdat = signal_normalized(file_path,mov_avg_windowSize=ws,sp=sop,xfact=1e6,xText=xText,skRows=skRows,
                                              titleText='full experiment',n_refs=n_refs,n_signals=n_signals,reduced_contrast=reduced_contrast)
        ## Normalized data 
        if i in flip_signal: # we need to flip the signal when 'phase_readout'=+90 
            ndata_0_list.append(1-ndat[1])
            ndata_1_list.append(1-ndat[3])
        else:
            ndata_0_list.append(ndat[1])
            ndata_1_list.append(ndat[3])
        errdata_0_list.append(errdat[1])
        errdata_1_list.append(errdat[3])
        #
        nv_short_0_list.append(ndat[0])
        nv_short_1_list.append(ndat[2])
        nv_shortE_0_list.append(errdat[0])
        nv_shortE_1_list.append(errdat[2])

    if zero_initial_time_u:
        xdat0=xdat0-xdat0[0]
        
    if cutoff is None:
        cutoff = len(xdat0) # do not remove any data
    xdat0 = xdat0[:cutoff]
    nd_re_1 = np.mean(np.array(ndata_0_list),axis=0)[:cutoff]
    ed_re_1 = np.sqrt(np.sum(np.array(errdata_0_list)**2,axis=0))[:cutoff]/len(file_list)
    nd_im_1 = np.mean(np.array(ndata_1_list),axis=0)[:cutoff]
    ed_im_1 = np.sqrt(np.sum(np.array(errdata_1_list)**2,axis=0))[:cutoff]/len(file_list)

    nv_short_re_1 = np.mean(np.array(nv_short_0_list),axis=0)[:cutoff]
    nv_shortE_re_1 = np.sqrt(np.sum(np.array(nv_shortE_0_list)**2,axis=0))[:cutoff]/(2*len(file_list))
    nv_short_im_1 = np.mean(np.array(nv_short_1_list),axis=0)[:cutoff]
    nv_shortE_im_1 = np.sqrt(np.sum(np.array(nv_shortE_1_list)**2,axis=0))[:cutoff]/(2*len(file_list))
    
    
    if return_norm_data==0:
        return nd_re_1,ed_re_1,nd_im_1,ed_im_1,nv_short_re_1,nv_shortE_re_1,nv_short_im_1,nv_shortE_im_1
    
    if sop:
        f0,fft0 = fftaux(xdat0,nd_re_1,add0s=True,sP=False)
        f1,fft1 = fftaux(xdat0,nd_im_1,add0s=True,sP=False)

        plt.plot(f0[:-1],fft0[:-1],label='experiment 0')
        plt.legend(loc=0);plt.show()
        print('max exp:',f0[:-1][fft0[:-1].argmax()])

        plt.plot(f1[:-1],fft1[:-1],label='experiment 1')
        plt.legend(loc=0);plt.show()
        print('max exp:',f1[:-1][fft1[:-1].argmax()])
    
    if ignore_y0a0:
        offset = 0
        factor = 1
    else:
        offset = y0
        factor = a0
    
    ##########################################################################################
    ##
    ## Add the (-u) part by 'reflecting' the data
    ##
    u_full_experim = np.concatenate([xdat0,-np.flip(xdat0[1:])])
    C_list_real   = +(1-2*(nd_re_1-offset)*factor) #+(2*(nd_re_1-offset)*factor-1) #
    C_list_real_E = abs(2*ed_re_1*factor)
    #C_list_imag   = ((-1)**int(add_minus_im))*(2*(nd_im_1-offset)*factor-1) 
    C_list_imag   = ((-1)**int(add_minus_im))*(1-2*(nd_im_1-0)*1) #((-1)**int(add_minus_im))*(2*(nd_im_1-0)*1-1) #
    # we do -(2*nd_im_1-1) only when the experimenal 'length_rf_half_pi_pulse' is a 2π multiple of 1/A
    C_list_imag_E = abs(2*ed_im_1*factor)
    C_list_bis_real   = np.concatenate([C_list_real,np.flip(C_list_real[1:])])
    C_list_bis_imag_m = np.concatenate([C_list_imag,-np.flip(C_list_imag[1:])]) # notice the (-) here
    C_list_bis_imag_p = np.concatenate([C_list_imag,+np.flip(C_list_imag[1:])]) # notice the (+) here
    C_list_bis_real_E = np.concatenate([C_list_real_E,np.flip(C_list_real_E[1:])])
    C_list_bis_imag_E = np.concatenate([C_list_imag_E,np.flip(C_list_imag_E[1:])])
    # the (-) or (+) signs on C(-u) are to obtain the real and imaginary part of the P(w), respectively
    
    if return_norm_data==1:
        return u_full_experim,C_list_bis_real,C_list_bis_imag_m,C_list_bis_imag_p,C_list_bis_real_E,C_list_bis_imag_E
            
    mask_exper_u = u_full_experim.argsort()
    #mask_sim = u_sim.argsort()
    if sop:
        for jj in range(2):
            C_list_bis_imag = [C_list_bis_imag_m,C_list_bis_imag_p][jj]
            print('C(-u)={}C(u)'.format(['-','+'][jj]))
            plt.errorbar(u_full_experim[mask_exper_u],C_list_bis_real[mask_exper_u],yerr=C_list_bis_real_E[mask_exper_u],
                         fmt='C0.:',label='Re[C]')
            plt.errorbar(u_full_experim[mask_exper_u],C_list_bis_imag[mask_exper_u],yerr=C_list_bis_imag_E[mask_exper_u],
                         fmt='C1.:',label='Im[C]')
            #plt.plot(u_sim[mask_sim],C_sim_real[mask_sim],'C0-',label='Re[C] (sim)')
            #plt.plot(u_sim[mask_sim],C_sim_imag[mask_sim],'C1-',label='Im[C] (sim)')
            
            plt.xlabel(xText,fontsize=12)
            plt.ylabel('Normalize data to |0,1> and |-1,1>',fontsize=10)
            #plt.ylim(-0.1,1.1)
            plt.axhline(0,ls='--',color='k') #,label='$|-1,1>$'
            plt.legend(loc=0,fontsize=10,ncol=2)
            plt.show()
    
    if use_IFFT: # Option of using the IFFT to get the KDQ from the characteristic function    
        ##########################################################################################
        ##
        ## Do the IFFT
        ##
        step = (u_full_experim[1:]-u_full_experim[:-1])[0]
        freqs_exper = np.fft.fftfreq(len(u_full_experim), step/np.pi)  # generate freq array 
        mask_exper = freqs_exper.argsort()
        
        pw_exper_R = np.fft.ifft(C_list_bis_real+1j*C_list_bis_imag_m) # notice the (_m) here
        pw_exper_I = np.fft.ifft(C_list_bis_real+1j*C_list_bis_imag_p) # notice the (_p) here
    ####
    ####
    else:  # Option of using the FFT to get the KDQ from the characteristic function    
        ##########################################################################################
        ##
        ## Do the FFT
        ##
        step = (u_full_experim[1:]-u_full_experim[:-1])[0]
        freqs_exper = np.fft.fftfreq(len(u_full_experim), step/np.pi)  # generate freq array 
        freqs_exper = - freqs_exper # flip sign to correct that we are doing FFT instead of IFFT
        mask_exper = freqs_exper.argsort()
        
        pw_exper_R = np.fft.fft(C_list_bis_real+1j*C_list_bis_imag_m,norm='forward') # notice the (_m) here
        pw_exper_I = np.fft.fft(C_list_bis_real+1j*C_list_bis_imag_p,norm='forward') # notice the (_p) here

        
    #from https://stackoverflow.com/questions/27529166/calculate-uncertainty-in-fft-amplitude
    # The uncertainty of FFT is unc(X_0) = unc(X_1) = ... = unc(X_(N-1)) = sqrt(unc(x1)**2 + unc(x2)**2 + ...)
    pw_exper_E = np.sqrt(np.sum(C_list_bis_real_E**2+C_list_bis_imag_E**2))/len(C_list_bis_imag_E)
    
    # Considering that the FFT is a linear function, then the error of each point is the FFT of all the errors
    # (summed in quadrature)
    #pw_exper_E = np.sqrt(np.sum(np.fft.ifft(np.sqrt(C_list_bis_real_E**2+C_list_bis_imag_E**2))**2)).real
    
    # rearrange in the order of the frequency vector
    freqs_exper = freqs_exper[mask_exper]
    pw_re = pw_exper_R.real[mask_exper]
    pw_im = pw_exper_I.imag[mask_exper]
    return freqs_exper,pw_re,pw_im,pw_exper_E


##############################################
## Analyze simulation
## from scrpit: kdq_2Lvls_interferometric.py
##
def func_simulation_ifft(u_sim,solx,soly, return_sim = False):
#                         fact_τ, ideal_sim=False,
#                         ω1 = (2*np.pi)*2.165,fact_δ1_over_Ω1 = 1.5*np.sqrt((np.sqrt(5)-1)/2),
#                         #Ωn = 2*np.pi/(2*2*15.704),n_half_pi = 15.704,
#                         r0A = np.array([[0, 0], [0, 1]]),r0S = np.array([[1, 0], [0, 0]]) ):
    """
    return_sim :  True=return simulation with u and (-u) parts, False=return ifft of simulation
    """
#    ##########################################################################################
#    ##
#    ## run simulation
#    ##
#    
#    # Initial state
#    #r0A = np.array([[0, 0], [0, 1]]) # oo = |0X0|_e (Note: for the nuclear spin it is inverted: oo = |1X1|_n)
#    #r0S = np.array([[1, 0], [0, 0]]) 
#    r0 = np.kron(r0S, r0A)
#
#    # Hamiltonian parameters in terms of the hyperfine coupling
#    #ω1 = (2*np.pi)*2.165 #define the oscillation in terms of the hyperfine
#    #fact_δ1_over_Ω1 = 1.5*np.sqrt((np.sqrt(5)-1)/2) #np.sqrt((np.sqrt(5)-1)/2) # 1 #
#    Ω1 = ω1/np.sqrt((fact_δ1_over_Ω1)**2 + 1) 
#    δ1 = Ω1 * fact_δ1_over_Ω1
#    if not(np.isclose(ω1 , np.sqrt(Ω1**2+δ1**2))):
#        print('Error: ω1 not equal to sqrt(Ω1^2 + δ1^2) ')
#
#    #θ = np.arctan(Ω1/δ1) 
#    # Due to issues with the high power (short mw pulses), the θ rotation is almost equal to zero
#    θ = np.arctan(Ω1/δ1) * 1.0 #
#    θ2 = θ + np.pi
#
#    tauExp_high = 0.025 #0.05 # µs
#    Ω_high = 2*np.pi/tauExp_high/2 # MHz # 3.8408
#    detu = -abs(A_phi)*(1.0) #*(0.0) # 1/2 # MHz detuning high power gates
#    # Note: detuning = -A means that the on-resonance electronic transition is the one for ms=+1 
#
#    # Factor that determines tau as a function of a pi rotation: 1 --> pi pulse, 2 --> 2 pi pulse, etc.
#    #fact_τ = 1 * 19/24
#
#    #Ωn = 2*np.pi/(2*30.765) #2*np.pi/(2*32.471) #
#    #n_half_pi = 15.242 #15.704 #16.166  #
#
#    if ideal_sim:
#        solx,soly = ideal_experiment(u_sim,r0,θ,θ2,x_gate_angle=(2-fact_τ)*np.pi,δ_high=detu,A=abs(A))
#    else:
#    	n_half_pi = 15.704 #
#    	solx,soly_bad = full_experiment(u_sim,r0,θ,θ2,Ω_high=Ω_high,δ_high=detu,A=abs(A),
#    	    	                        n_half_pi_length=n_half_pi,#Ω_n=Ωn,
#                    	     	        do_spin_echo_2=True,#pulse1_angle=0,add_dephasing=True,
#                                	x_gate_angle=(fact_τ)*np.pi,ideal_readout=False,phi_factor=1,phi_factor2=1)
#    	n_half_pi = 16.166 #
#    	solx_bad,soly = full_experiment(u_sim,r0,θ,θ2,Ω_high=Ω_high,δ_high=detu,A=abs(A),
#            	                        n_half_pi_length=n_half_pi,#Ω_n=Ωn,
#            	                        do_spin_echo_2=True,pulse1_angle=0,#add_dephasing=True,
#            	                        x_gate_angle=(fact_τ)*np.pi,ideal_readout=False,phi_factor=1,phi_factor2=1)
#        #solx_bad,soly = full_experiment(u_sim,r0,θ,θ2,Ω_high=Ω_high,δ_high=detu,#Ω_n=Ωn,n_half_pi_length=n_half_pi,
#        #                                do_spin_echo_2=False,
#        #                                x_gate_angle=(2-fact_τ)*np.pi,ideal_readout=False,phi_factor=1)
#        #solx,soly_bad = full_experiment(u_sim,r0,θ,θ2,Ω_high=Ω_high,δ_high=detu,#Ω_n=Ωn,n_half_pi_length=n_half_pi,
#        #                                do_spin_echo_2=True,pulse1_angle=0,
#        #                                x_gate_angle=(2-fact_τ)*np.pi,ideal_readout=False,phi_factor=1)
#    #print(np.allclose(solx.imag,0),np.allclose(soly.imag,0))
#    solx,soly = solx.real,soly.real
#    if return_sim==0:
#        return solx,soly
    
    
    ##########################################################################################
    ##
    ## Add the (-u) part by 'reflecting' the data 
    ##
    u_sim_2 = np.concatenate([u_sim,-np.flip(u_sim[1:])])
    C_sim_real   = 1-2*solx #2*solx-1 #
    C_sim_imag   = 1-2*soly #2*soly-1 #
    C_sim_real   = np.concatenate([C_sim_real,np.flip(C_sim_real[1:])])
    C_sim_imag_m = np.concatenate([C_sim_imag,-np.flip(C_sim_imag[1:])]) #-
    C_sim_imag_p = np.concatenate([C_sim_imag,+np.flip(C_sim_imag[1:])])
    # the (-) or (+) signs on C(-u) are to obtain the real and imaginary part of the P(w), respectively
    
    if return_sim:
        return u_sim_2,C_sim_real,C_sim_imag_m,C_sim_imag_p
    ##########################################################################################
    ##
    ## Do the IFFT
    ##
    step = (u_sim_2[1:]-u_sim_2[:-1])[0]
    freqs_sim = np.fft.fftfreq(len(u_sim_2), step/np.pi)  # generate freq array 
    mask_sim = freqs_sim.argsort()
    
    pw_sim_R = np.fft.ifft(C_sim_real+1j*C_sim_imag_m) # notice the (_m) here
    pw_sim_I = np.fft.ifft(C_sim_real+1j*C_sim_imag_p) # notice the (_p) here
    
    # rearrange in the order of the frequency vector
    freqs_sim = freqs_sim[mask_sim]
    pw_s_re = pw_sim_R.real[mask_sim]
    pw_s_im = pw_sim_I.imag[mask_sim]
    
    return freqs_sim,pw_s_re,pw_s_im
    
    
    ##############################################
## Analyze simulation
## from scrpit: Interferometric_scheme_self_consistent_IFFT.py
## in Documents/Postdoc/2022_Postdoc_MIT/2023_iPython/
##
def func_simulation_fft(u_sim,solx,soly, return_sim = False):
#    C_sim_real   = 1-2*solx #2*solx-1 #
#    C_sim_imag   = 1-2*soly #2*soly-1 #
#    ##########################################################################################
#    ##
#    ## Do the FFT
#    ##
#    step = (u_sim[1:]-u_sim[:-1])[0]
#    freqs_sim = np.fft.fftfreq(len(u_sim), step/np.pi)  # generate freq array 
#    freqs_sim = - freqs_sim # flip sign to correct that we are doing FFT instead of IFFT
#    mask_sim = freqs_sim.argsort()
#    
#    pw_sim = np.fft.fft(C_sim_real+1j*C_sim_imag)
#    pw_sim = pw_sim/len(solx) #renormalize
#    
#    # rearrange in the order of the frequency vector
#    freqs_sim = freqs_sim[mask_sim]
#    pw_s_re = pw_sim.real[mask_sim]
#    pw_s_im = pw_sim.imag[mask_sim]
#    
#    return freqs_sim,pw_s_re,pw_s_im
    u_sim = np.concatenate([u_sim,-np.flip(u_sim[1:])])
    C_sim_real = 1-2*solx #2*solx-1 #
    C_sim_imag = 1-2*soly #2*soly-1 #
    C_sim_real = np.concatenate([C_sim_real,np.flip(C_sim_real[1:])])
    C_sim_imag_m = np.concatenate([C_sim_imag,-np.flip(C_sim_imag[1:])]) #-
    C_sim_imag_p = np.concatenate([C_sim_imag,+np.flip(C_sim_imag[1:])])

    if return_sim:
        return u_sim,C_sim_real,C_sim_imag_m,C_sim_imag_p
    ##########################################################################################
    ##
    ## Do the FFT
    ##
    step = (u_sim[1:]-u_sim[:-1])[0]
    freqs_sim = np.fft.fftfreq(len(u_sim), step/np.pi)  # generate freq array 
    freqs_sim = - freqs_sim # flip sign to correct that we are doing FFT instead of IFFT
    mask_sim = freqs_sim.argsort()
    
    pw_sim_R = np.fft.fft(C_sim_real+1j*C_sim_imag_m) # notice the (_m) here
    pw_sim_I = np.fft.fft(C_sim_real+1j*C_sim_imag_p) # notice the (_p) here
    pw_sim_R = pw_sim_R/len(u_sim) #renormalize
    pw_sim_I = pw_sim_I/len(u_sim) #renormalize
    
    # rearrange in the order of the frequency vector
    freqs_sim = freqs_sim[mask_sim]
    pw_s_re = pw_sim_R.real[mask_sim]
    pw_s_im = pw_sim_I.imag[mask_sim]
    
    return freqs_sim,pw_s_re,pw_s_im
