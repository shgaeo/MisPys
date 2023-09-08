# Script to calculate the Kirkwood-Dirac quasiprobability distribution for a 2 level system under unitary evolution for a time varying Hamiltonian
# H(t) = (1/2)*(Ω*(np.cos(t*δ) \sigma_x + np.sin(t*δ) \sigma_y ) + \delta \sigma_z)

import numpy as np


# Definition of auxiliary matrices 
id2=np.array([[1,0],[0,1]])
σx = np.array([[0,1],[1,0]])
σy = np.array([[0,-1j],[1j,0]])
σz = np.array([[1,0],[0,-1]])


def get_time_list(Ω,n_periods,n_points):
    return np.linspace(0,n_periods*2*np.pi/Ω,n_points)

def get_pϕ_from_rho(ρ):
    """
    takes a pure state density matrix ρ=[[p,sqrt{p}sqrt{1-p}exp(i phi)],[sqrt{p}sqrt{1-p}exp(-i phi),1-p]]")
    and returns the parameters p,phi
    """
    phi = (np.log(ρ[0,1]/ρ[1,0])/2).imag
    p = ρ[0,0].real
    #print(np.array([[p,np.sqrt(p)*np.sqrt(1-p)*np.exp(1j*phi)],
    #                [np.sqrt(p)*np.sqrt(1-p)*np.exp(-1j*phi),1-p]]))
    return p,phi

def get_ket_from_rho(ρ):
    """
    takes a pure state density matrix ρ=[[p,sqrt{p}sqrt{1-p}exp(i phi)],[sqrt{p}sqrt{1-p}exp(-i phi),1-p]]")
    and returns the ket = [sqrt{p} , sqrt{1-p}e^{i phi}]
    """
    p,phi = get_pϕ_from_rho(ρ)
    if np.isclose(phi,0):
        return np.sqrt(p) , np.sqrt(1-p)
    return np.sqrt(p) , np.sqrt(1-p)*np.exp(1j*phi)
    
def get_KDQ_fast(xp,n_points=50,n_periods=1,init_state='th',check_Ht_evecs=False,return_joint_probs=False,β_fact=1):
    Ω,δ = xp
    ω = np.sqrt(δ**2 + Ω**2)

    tlist = get_time_list(Ω,n_periods,n_points)
    
    # H(t=0) = (1/2)*(Ω*\sigma_x + \delta \sigma_z)
    # (see /home/santiago/Documentos/Escuela-doctorado/roba_di_mathematica/2Lvls_unitary_time_varying_H.nb)
    rho0_up = np.array([[ (1 + δ/ω),  Ω/ω], [ Ω/ω, 1 - δ/ω]])/2
    rho0_do = np.array([[ (1 - δ/ω), -Ω/ω], [-Ω/ω, 1 + δ/ω]])/2
    # Initial state |ψ> = (|↑> - |↓>)/sqrt(2)
    rho0_mi = np.array([[ (1 + Ω/ω), -δ/ω], [-δ/ω, 1 - Ω/ω]])/2
    # Initial state |ψ> = (|↑> + |↓>)/sqrt(2)
    rho0_pl = np.array([[ (1 - Ω/ω),  δ/ω], [ δ/ω, 1 + Ω/ω]])/2
    
    
    # Defining unitary operator as U = exp(-iδ/2 σ_z)exp(-iΩ/2 σ_x)
    def uU(t):
        real_part =  np.cos(δ*t/2)*np.cos(Ω*t/2)*id2 - np.sin(δ*t/2)*np.sin(Ω*t/2)*1j*σy
        imag_part = -np.sin(δ*t/2)*np.cos(Ω*t/2)*σz  - np.cos(δ*t/2)*np.sin(Ω*t/2)*σx
        return real_part+1j*imag_part
    #
    # Define initial state
    if init_state == 'u':
        rho_0 = rho0_up
    elif init_state == 'd':
        rho_0 = rho0_do
    elif init_state == 'p':
        rho_0 = rho0_pl
    elif init_state == 'm':
        rho_0 = rho0_mi
    elif init_state == 'th':
        β = β_fact/ω #1/2.2/2/np.pi #1/kHz
        zβ = np.exp(β*ω/2) + np.exp(-β*ω/2)
        rho_0 = rho0_do*np.exp(-β*ω/2)/zβ + rho0_up*np.exp(β*ω/2)/zβ
    elif init_state == '0':
        rho_0 = np.array([[ 1, 0], [0, 0]])
    elif init_state == '1':
        rho_0 = np.array([[ 0, 0], [0, 1]])
    elif init_state[:4] == 'pure':
        p_eff,phi_eff = float(init_state.split(',')[1]),float(init_state.split(',')[2])
        rho_0 = np.array([[p_eff,np.sqrt(p_eff)*np.sqrt(1-p_eff)*np.exp(1j*phi_eff)],
                          [np.sqrt(p_eff)*np.sqrt(1-p_eff)*np.exp(-1j*phi_eff),1-p_eff]])
    else:
        print("Error: init_state is not one of the following:")
        print("init_state = 'u','d','p','m','th','0','1' for states \ket{up},\ket{down},\ket{+},\ket{-},\rho_thermal,\ket{0},\ket{1} respectively")
        return -1
    
    
    q_fi = np.zeros([len(tlist),2,2],dtype=complex)
    p_tpm_fi = np.zeros([len(tlist),2,2],dtype=complex)
    
    for i,t in enumerate(tlist):
        # total propagator:
        opU   = uU(t)
        opUct = opU.conj().transpose()
        
        # Hamiltonian instantaneous eigenvalues H(t) = (1/2)*(Ω*(np.cos(t*δ) \sigma_x + np.sin(t*δ) \sigma_y ) + \delta \sigma_z)
        # (see /home/santiago/Documentos/Escuela-doctorado/roba_di_mathematica/2Lvls_unitary_time_varying_H.nb)
        rhot_up = np.array([[ (1 + δ/ω),  np.exp(-1j*t*δ)*Ω/ω], [ np.exp(1j*t*δ)*Ω/ω, 1 - δ/ω]])/2
        rhot_do = np.array([[ (1 - δ/ω), -np.exp(-1j*t*δ)*Ω/ω], [-np.exp(1j*t*δ)*Ω/ω, 1 + δ/ω]])/2
        if check_Ht_evecs:
            Ht = np.array([[δ,Ω*(np.cos(t*δ)-1j*np.sin(t*δ))],[Ω*(np.cos(t*δ)+1j*np.sin(t*δ)),-δ]])/2
            if False in [np.allclose(np.dot(Ht,rhot_up),ω*rhot_up/2),
                         np.allclose(np.dot(Ht,rhot_do),-ω*rhot_do/2)]:
                print('Something is wrong with the Hamiltonian eigenstates at t='+str(t))
        # Calculate the KDQ
        # first the part without the initial Hamiltonian projector (to only calculate this once)
        #Ut_Xido_U_rho = np.dot(np.dot(np.dot(opUct,rhot_do),opU),rho_0)
        #Ut_Xiup_U_rho = np.dot(np.dot(np.dot(opUct,rhot_up),opU),rho_0)
        #q_fi[i,0,0] = np.dot(rho0_up,Ut_Xiup_U_rho).trace()
        #q_fi[i,0,1] = np.dot(rho0_up,Ut_Xido_U_rho).trace()
        #q_fi[i,1,0] = np.dot(rho0_do,Ut_Xiup_U_rho).trace()
        #q_fi[i,1,1] = np.dot(rho0_do,Ut_Xido_U_rho).trace()    
        q_fi[i,0,0] = np.dot(np.dot(np.dot(np.dot(opUct,rhot_up),opU),rho0_up),rho_0).trace()
        q_fi[i,0,1] = np.dot(np.dot(np.dot(np.dot(opUct,rhot_do),opU),rho0_up),rho_0).trace()
        q_fi[i,1,0] = np.dot(np.dot(np.dot(np.dot(opUct,rhot_up),opU),rho0_do),rho_0).trace()
        q_fi[i,1,1] = np.dot(np.dot(np.dot(np.dot(opUct,rhot_do),opU),rho0_do),rho_0).trace()    
        if return_joint_probs:
            # Calculate the TPM probs
            U_Pido_Ut = np.dot(np.dot(opU,rho0_do),opUct)
            U_Piup_Ut = np.dot(np.dot(opU,rho0_up),opUct)
            p_tpm_fi[i,0,0] = np.dot(rhot_up,U_Piup_Ut).trace() * np.dot(rho_0,rho0_up).trace()
            p_tpm_fi[i,1,0] = np.dot(rhot_up,U_Pido_Ut).trace() * np.dot(rho_0,rho0_do).trace()
            p_tpm_fi[i,0,1] = np.dot(rhot_do,U_Piup_Ut).trace() * np.dot(rho_0,rho0_up).trace()
            p_tpm_fi[i,1,1] = np.dot(rhot_do,U_Pido_Ut).trace() * np.dot(rho_0,rho0_do).trace()
            
    if return_joint_probs:
        return q_fi,p_tpm_fi
    return q_fi

def get_charFunc_fast(xp,u_vec,tau,init_state='th',check_Ht_evecs=False,β_fact=1):
    Ω,δ = xp
    ω = np.sqrt(δ**2 + Ω**2)
    
    # H(t=0) = (1/2)*(Ω*\sigma_x + \delta \sigma_z)
    # (see /home/santiago/Documentos/Escuela-doctorado/roba_di_mathematica/2Lvls_unitary_time_varying_H.nb)
    rho0_up = np.array([[ (1 + δ/ω),  Ω/ω], [ Ω/ω, 1 - δ/ω]])/2
    rho0_do = np.array([[ (1 - δ/ω), -Ω/ω], [-Ω/ω, 1 + δ/ω]])/2
    # Initial state |ψ> = (|↑> - |↓>)/sqrt(2)
    rho0_mi = np.array([[ (1 + Ω/ω), -δ/ω], [-δ/ω, 1 - Ω/ω]])/2
    # Initial state |ψ> = (|↑> + |↓>)/sqrt(2)
    rho0_pl = np.array([[ (1 - Ω/ω),  δ/ω], [ δ/ω, 1 + Ω/ω]])/2
    # Define initial state
    if init_state == 'u':
        rho_0 = rho0_up
    elif init_state == 'd':
        rho_0 = rho0_do
    elif init_state == 'p':
        rho_0 = rho0_pl
    elif init_state == 'm':
        rho_0 = rho0_mi
    elif init_state == 'th':
        β = β_fact/ω #1/2.2/2/np.pi #1/kHz
        zβ = np.exp(β*ω/2) + np.exp(-β*ω/2)
        rho_0 = rho0_do*np.exp(-β*ω/2)/zβ + rho0_up*np.exp(β*ω/2)/zβ
    elif init_state == '0':
        rho_0 = np.array([[ 1, 0], [0, 0]])
    elif init_state == '1':
        rho_0 = np.array([[ 0, 0], [0, 1]])
    elif init_state[:4] == 'pure':
        p_eff,phi_eff = float(init_state.split(',')[1]),float(init_state.split(',')[2])
        rho_0 = np.array([[p_eff,np.sqrt(p_eff)*np.sqrt(1-p_eff)*np.exp(1j*phi_eff)],
                          [np.sqrt(p_eff)*np.sqrt(1-p_eff)*np.exp(-1j*phi_eff),1-p_eff]])
    else:
        print("Error: init_state is not one of the following:")
        print("init_state = 'u','d','p','m','th','0','1' for states \ket{up},\ket{down},\ket{+},\ket{-},\rho_thermal,\ket{0},\ket{1} respectively")
        print("or init_state = 'pure,'+str(p)+','+str(phi) for a pure state ρ=[[p,sqrt{p}sqrt{1-p}exp(i phi)],[sqrt{p}sqrt{1-p}exp(-i phi),1-p]]")
        return -1
    
    # Defining unitary operator as U = exp(-iδ/2 σ_z)exp(-iΩ/2 σ_x)
    def uU(t):
        real_part =  np.cos(δ*t/2)*np.cos(Ω*t/2)*id2 - np.sin(δ*t/2)*np.sin(Ω*t/2)*1j*σy
        imag_part = -np.sin(δ*t/2)*np.cos(Ω*t/2)*σz  - np.cos(δ*t/2)*np.sin(Ω*t/2)*σx
        return real_part+1j*imag_part
    # total propagator:
    opU   = uU(tau)
    opUct = opU.conj().transpose()
    if check_Ht_evecs:
        # Hamiltonian instantaneous eigenvalues H(t) = (1/2)*(Ω*(np.cos(t*δ) \sigma_x + np.sin(t*δ) \sigma_y ) + \delta \sigma_z)
        # (see /home/santiago/Documentos/Escuela-doctorado/roba_di_mathematica/2Lvls_unitary_time_varying_H.nb)
        rhot_up = np.array([[ (1 + δ/ω),  np.exp(-1j*tau*δ)*Ω/ω], [ np.exp(1j*tau*δ)*Ω/ω, 1 - δ/ω]])/2
        rhot_do = np.array([[ (1 - δ/ω), -np.exp(-1j*tau*δ)*Ω/ω], [-np.exp(1j*tau*δ)*Ω/ω, 1 + δ/ω]])/2
        Ht = np.array([[δ,Ω*(np.cos(tau*δ)-1j*np.sin(tau*δ))],[Ω*(np.cos(tau*δ)+1j*np.sin(tau*δ)),-δ]])/2
        if False in [np.allclose(np.dot(Ht,rhot_up),ω*rhot_up/2),
                     np.allclose(np.dot(Ht,rhot_do),-ω*rhot_do/2)]:
            print('Something is wrong with the Hamiltonian eigenstates at tau='+str(tau))
        
    Gu = np.zeros(len(u_vec),dtype=complex)
    for i,u in enumerate(u_vec):
        # Unitary propagators for time u
        exp_miH0 = np.cos(u*ω/2)*id2 - 1j*np.sin(u*ω/2)*( (Ω/ω)*σx + (δ/ω)*σz ) # exp(-i u H(0))
        exp_plHt = np.cos(u*ω/2)*id2 + 1j*np.sin(u*ω/2)*( (Ω/ω)*(np.cos(tau*δ)*σx + np.sin(tau*δ)*σy ) + (δ/ω)*σz ) # exp( i u H(t))
        
        # Calculate the real and imaginary parts of the characteristic function:
        # G(u) = Tr[ e^{-iuH0} ρ U^† e^{iuH0} U ]
        Gu[i] = np.dot(np.dot(np.dot(np.dot(exp_miH0,rho_0),opUct),exp_plHt),opU).trace()
        
    return Gu


##
## Analytic expression for the real part:
##
# The real part of the characteristic function is: Re$[G(u)] = (G+G^*)/2 = {\rm Tr} [\rho (U^\dagger e^{(+i u H_\tau)} U e^{(-i u H_0)} + e^{(+i u H_0)} U^\dagger e^{(-i u H_\tau)} U)]/2 $, 
# In /home/santiago/Documents/Postdoc/2022_Postdoc_MIT/Roba di Mathematica/Interferometric_FullExperiment_Hyperfine.nb we show that 
# $U^\dagger e^{(+i u H_\tau)} U e^{(-i u H_0)} + e^{(+i u H_0)} U^\dagger e^{(-i u H_\tau)} U  = 2 k \mathcal{I}$
# where 
# $k = 1 - 2\frac{\delta^2}{\omega^2} \sin^2\frac{\tau\Omega}{2} \; \sin^2\frac{u \omega}{2}$
# for $\omega = \sqrt{\Omega^2 + \delta^2}$
# 
# This means that Re$[G(u)] = {\rm Tr} [\rho \mathcal{I} 2k]/2$, i.e.,
# ${\rm Re}[G(u)] = k$
#
# NOTE: this is only valid for our Hamiltonian H(t) = (1/2)*(Ω*(np.cos(t*δ) \sigma_x + np.sin(t*δ) \sigma_y ) + \delta \sigma_z). For which the unitary evolution is U = exp(-iδ/2 σ_z)exp(-iΩ/2 σ_x)
#
def func_reGu(xp,u_list,tau):
    Ω,δ = xp
    ω = np.sqrt(δ**2 + Ω**2)
    #return 1 + (δ/ω)**2 * (np.cos(tau*Ω)-1)*(np.sin(ω*u_list/2)**2)
    return 1 - 2*(δ/ω)**2 * (np.sin(tau*Ω/2)**2)*(np.sin(ω*u_list/2)**2)


##
## Simulation of the interferometric scheme
## (scheme that considers the hyperfine terms)
##
# Definitions
oo = np.array([[0, 0], [0, 1]]) # oo = |0X0|_e (Note: for the nuclear spin it is inverted: oo = |1X1|_n)
zz = np.array([[1, 0], [0, 0]]) # zz = |1X1|_e (Note: for the nuclear spin it is inverted: zz = |0X0|_n)
zo = np.array([[0, 1], [0, 0]]) # zo = |0X1|_e (Note: for the nuclear spin it is inverted: zo = |1X0|_n)
oz = np.array([[0, 0], [1, 0]]) # oz = |1X0|_e (Note: for the nuclear spin it is inverted: oz = |0X1|_n)
dims = 2
id2 = σz**2

#A = -2.1875 * 2*np.pi #
#A = -2.165 * 2*np.pi # MHz #2.15
A = -2.1611 * 2*np.pi # MHz # See Documents/Postdoc/2022_Postdoc_MIT/2023_iPython/2023_08/2023_08_11_N_ramsey_calibrate_half_pi_pulses.ipynb


tauExp_low = 0.400 #0.942 # µs 0.043 #
# 400ns means that a pi on resonance for m_I=+1 will be a 2pi for m_I=0
Ω_low = 2*np.pi/tauExp_low/2 # MHz # 3.8408
δ_low = Ω_low*1.5*np.sqrt((np.sqrt(5) - 1)/2) # MHz
ω_low = np.sqrt(Ω_low**2 + δ_low**2)

tauExp_high = 0.025 #0.05 # µs
Ω_high = 2*np.pi/tauExp_high/2 # MHz # 3.8408
δ_high = 0 #A/2 # MHz
#
#τ = (1/2)*(2*np.pi)/Ω_low # half period ; µs 

Ω_n = 2*np.pi/31.545/2 # MHz # 3.8408
δ_n = 0*A # MHz
fact_n = 15.741/(31.545/2) #1 # extra factor to make A*pi = 2*pi
n_pi_2_length = (np.pi/2)*fact_n*(1/Ω_n)
n_pi_2_length == 15.741 #15.277 #15.741 #


