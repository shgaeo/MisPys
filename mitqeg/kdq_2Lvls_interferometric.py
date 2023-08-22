# Script to calculate the Kirkwood-Dirac quasiprobability distribution for a 2 level system under unitary evolution for a time varying Hamiltonian
# H(t) = (1/2)*(Ω*(np.cos(t*δ) \sigma_x + np.sin(t*δ) \sigma_y ) + \delta \sigma_z)

import numpy as np
from scipy.linalg import expm

# This code is based on the notebook:
# /home/santiago/Documents/Postdoc/2022_Postdoc_MIT/2023_iPython/N_ramsey_Characteristic_function_Simulations_of_the_full_experimen_including_hyperfine_coupling.ipynb

###########################################
## Simulation of the interferometric scheme
## (scheme that considers the hyperfine terms)
##

##
## Definition of auxiliary matrices
##
id2=np.array([[1,0],[0,1]])
σx = np.array([[0,1],[1,0]])
σy = np.array([[0,-1j],[1j,0]])
σz = np.array([[1,0],[0,-1]])

##
## Definition of hyperfine coupling
##
#A = -2.1875 * 2*np.pi #
#A = -2.165 * 2*np.pi # MHz #2.15
A = -2.1611 * 2*np.pi # MHz # See Documents/Postdoc/2022_Postdoc_MIT/2023_iPython/2023_08/2023_08_11_N_ramsey_calibrate_half_pi_pulses.ipynb
#
# In addition, we need to distinguish between the hyperfine A and the Azz set in the experiment 
# that defines the phase of the RF pulses. Up to 2023-08-18 it was 2.165 MHz or 2.16 MHz
A_phi = 2.165*2*np.pi
# Notice also here the positive sign. This is to make simulation equal to the experiment

##
## Definition of parameters (can be also specified when calling the functions later on)
##
tauExp_low = 0.400 #0.942 # µs 0.043 #
# 400ns means that a pi on resonance for m_I=+1 will be a 2pi for m_I=0
Ω_low = 2*np.pi/tauExp_low/2 # MHz # 3.8408
#δ_low = Ω_low*1.5*np.sqrt((np.sqrt(5) - 1)/2) # MHz
#ω_low = np.sqrt(Ω_low**2 + δ_low**2)
#
tauExp_high = 0.025 #0.05 # µs
Ω_high = 2*np.pi/tauExp_high/2 # MHz # 3.8408
δ_high = 0 #A/2 # MHz
#
#τ = (1/2)*(2*np.pi)/Ω_low # half period ; µs
#
Ω_n = 2*np.pi/31.545/2 # MHz # 3.8408
δ_n = 0*A # MHz
fact_n = 15.741/(31.545/2) #1 # extra factor to make (approximately) A*pi = 2*pi
n_half_pi_length = (np.pi/2)*fact_n*(1/Ω_n)
# for reference:
# n_half_pi_length == 15.741 and -2*np.pi/A*34 == 15.740740740740739

##
## Definition of a dephasing matrix (full dephasing)
## To do: for less strong dephasing, add some exp decay instead of zeros
##
dephasing = np.array([[1, 1, 0, 0],
                      [1, 1, 0, 0],
                      [0, 0, 1, 1],
                      [0, 0, 1, 1]])
# note that this dephasing matrix is meant to multiply rho element-by-element. Not with the usual matrix product


##
## Definition of the system+ancilla Hamiltonian including the hyperfine term
##
# Driven qubit Hamiltonian:
def hamilt_qubit(Ω_p,δ_p,ph_p):
    return (1/2)*(Ω_p*(np.cos(ph_p)*σx + np.sin(ph_p)*σy) + δ_p*σz)
# NV+N Hamiltonian (for |m_s,m_i>)
def hamilt_H(Ω_e,δ_e,ph_e,Ω_n,δ_n,ph_n,Azz):
    hS = np.kron(hamilt_qubit(Ω_e,δ_e,ph_e),id2) # System Hamiltonian
    hA = np.kron(id2,hamilt_qubit(Ω_n,δ_n,ph_n)) # Ancilla Hamiltonian
    hh = (Azz/4)*(np.kron(σz,σz) + np.kron(σz,id2) - np.kron(id2,σz) - np.kron(id2,id2)) # Hyperfine term
    return hS+hA+hh
# for reference:
# hamilt_H = (1/2)[[ δe+δn,      Ωn,        Ωe,      0 ],
#                  [    Ωn,   δe-δn,         0,     Ωe ]
#                  [    Ωe,       0, -2A-δe+δn,     Ωn ]
#                  [     0,      Ωe,        Ωn, -δe-δn ]]


##
## Definition of auxiliary functions
##
# Partial traces
def traceA(ρ):
    #return np.array([[ρ[0,0] + ρ[1,1], ρ[2,0] + ρ[3,1]], [ρ[0,2] + ρ[1,3], ρ[2,2] + ρ[3,3]]])
    return np.array([[ρ[0,0] + ρ[1,1], ρ[0,2] + ρ[1,3]], [ρ[2,0] + ρ[3,1], ρ[2,2] + ρ[3,3]]])
def traceS(ρ):
    #return np.array([[ρ[0,0] + ρ[2,2], ρ[1,0] + ρ[3,2]], [ρ[0,1] + ρ[2,3], ρ[1,1] + ρ[3,3]]])
    return np.array([[ρ[0,0] + ρ[2,2], ρ[0,1] + ρ[2,3]], [ρ[1,0] + ρ[3,2], ρ[1,1] + ρ[3,3]]])
# conjugate transpose and matrix multiplication (just to simplify notation)
    return np.matmul
def ct(x):
    return np.conjugate(np.transpose(x))
def mul(x,y):
    return np.matmul(x,y)



##
## Function to simulate the full experiment
##
def full_experiment(u_vec,r0,θ,θ2, x_gate_angle=np.pi,pulse1_angle=np.pi/2,Ω_high=Ω_high,δ_high=δ_high,Ω_n=Ω_n,
                    δ_n=δ_n,A=A,A_phi=A_phi,n_half_pi_length=n_half_pi_length,ideal_readout=False,phi_factor=1,phi_factor2=1,
                    spin_echo_phase=np.pi/2,spin_echo_angle=np.pi,do_spin_echo_0=False,do_spin_echo_1=True,do_spin_echo_2=False,return_S=False,
                    do_3rd_Rf=True,add_dephasing=False,short_laser=True,Γ_dephasing=None,do_CNOT=True,
                    add_C13=False,γC13=0):
    # Definition of nuclear gates
    #n_rympi2 = expm(-1j*(np.pi/2)*fact_n*(1/Ω_n)*hamilt_H(0,δ_high,0, Ω_n,δ_n,-np.pi/2, A)) # R_y(-π/2) = R_{-y}(π/2)
    n_rympi2 = expm(-1j*(n_half_pi_length)*hamilt_H(0,δ_high,0, Ω_n,δ_n,-(-np.pi/2), A)) # R_y(-π/2) = R_{-y}(π/2) # MINUS PHASE
    
    # Definition of electronic gates
    rypi2 = expm(-1j*(pulse1_angle)*(1/Ω_high)*hamilt_H(Ω_high,δ_high,np.pi/2, 0,δ_n,0, A)) # R_y(π/2)
    ryθ = expm(-1j*(θ)*(1/Ω_high)*hamilt_H(Ω_high,δ_high,np.pi/2, 0,δ_n,0, A)) # R_y(θ)
    #rymθ = expm(+1j*(θ)*(1/Ω_high)*hamilt_H(Ω_high,δ_high,np.pi/2, 0,δ_n,0, A)) # R_y(-θ)
    #rymθ2 = expm(+1j*(θ2)*(1/Ω_high)*hamilt_H(Ω_high,δ_high,np.pi/2, 0,δ_n,0, A)) # R_y(-θ2)     # This is not correct, you need to add/substract a pi to the phase to emulate -θ2.
                                                                                                # Otherwise you are also reversing the evolution of the hyperfine, which is obviusly not possible
    rymθ2 = expm(-1j*(θ2)*(1/Ω_high)*hamilt_H(Ω_high,δ_high,-np.pi/2, 0,δ_n,0, A)) # R_y(-θ2)     # This is the correct definition of the gate for -θ2.
    rxgate = expm(-1j*(x_gate_angle)*(1/Ω_high)*hamilt_H(Ω_high,δ_high,np.pi, 0,δ_n,0, A)) # R_x(π) ##Im adding a pi phase to the x gate that is not present in the experiment
    #rxpi = expm(-1j*(np.pi)*(1/Ω_high)*hamilt_H(Ω_high,δ_high,0, 0,δ_n,0, A)) # R_x(π)
    rxypi = expm(-1j*(spin_echo_angle)*(1/Ω_high)*hamilt_H(Ω_high,δ_high,spin_echo_phase, 0,δ_n,0, A)) # R_xy(π)  "spin echo 2"
    rxypi_2 = expm(-1j*(spin_echo_angle)*(1/Ω_high)*hamilt_H(Ω_high,δ_high,spin_echo_phase+np.pi, 0,δ_n,0, A)) # R_xy(π) "spin echo 2" has an extra pi phase (why? I don't know, but it was like this in the experiment)
    
    # low power gate (CNOT but with hyperfine)
    cnot_h = expm(-1j*(np.pi/Ω_low)*hamilt_H(Ω_low,δ_high,0, 0,δ_n,0, A)) # low power pi (same detuning as high power)
    
    #print(np.allclose(u_vec,xdat0))
    solx = np.zeros(len(u_vec),dtype=complex)
    soly = np.zeros(len(u_vec),dtype=complex)
    if return_S:
        rho_s_Re = np.zeros([len(u_vec),2,2],dtype=complex)
        rho_s_Im = np.zeros([len(u_vec),2,2],dtype=complex)

    # first pi/2 nuclear spin:
    r0 = mul(n_rympi2, mul(r0,ct(n_rympi2)))

    # first pi/2 e- spin:
    r1 = mul(rypi2, mul(r0,ct(rypi2)))

    #extra correction of phase due to evolution during high power gates (to be characterized)
    #t_extra = 0.0 #0.05 # 0.85
    #t_extra = (np.pi/2)*(1/Ω_high) + (θ)*(1/Ω_high) + (θ2)*(1/Ω_high) + (np.pi)*(1/Ω_high)
    #t_extra = (pulse1_angle)*(1/Ω_high) + (θ)*(1/Ω_high) + (θ2)*(1/Ω_high) + (x_gate_angle)*(1/Ω_high)
    t_extra = (pulse1_angle + do_spin_echo_0*np.pi)*(1/Ω_high) + (θ)*(1/Ω_high) + (θ2)*(1/Ω_high) + (x_gate_angle)*(1/Ω_high)
    t_extra2= (np.pi)*(1/Ω_high) + n_half_pi_length

    for i,tt in enumerate(u_vec):
        free = expm(-1j*tt*hamilt_H(0,δ_high,0, 0,δ_n,0, A)) # free evolution

        r2 = mul(free, mul(r1,ct(free)))
        #
        if add_C13:
            # this is to emulate the effective evolution induced by a nearby C-13 with parallel component γ13 (along z). 
            # For details see /home/santiago/Documents/Postdoc/2022_Postdoc_MIT/Roba di Mathematica/Hyperfine_NV_N_C13.nb 
            cosγC13 = np.cos(γC13*tt)
            c13Eff = np.array([[      1,       1, cosγC13, cosγC13], 
                               [      1,       1, cosγC13, cosγC13],
                               [cosγC13, cosγC13,       1,       1],
                               [cosγC13, cosγC13,       1,       1]])
            r2 = r2*c13Eff # note element-by-element product
        #
        if add_dephasing:
            if Γ_dephasing is None:
                r2 = r2*dephasing # note element-by-element product
            else:
                eΓt = np.exp(-Γ_dephasing*tt)
                dephasing_tt = np.array([[1,     1, eΓt, eΓt],
                                         [1,     1, eΓt, eΓt],
                                         [eΓt, eΓt,   1,   1],
                                         [eΓt, eΓt,   1,   1]])
                r2 = r2*dephasing_tt # note element-by-element product

        r3 = mul(ryθ,  mul(r2,ct(ryθ)))
        r4 = mul(rxgate, mul(r3,ct(rxgate)))
        #r5 = mul(rymθ, mul(r4,ct(rymθ)))
        r5 = mul(rymθ2, mul(r4,ct(rymθ2)))
        r6 = mul(free, mul(r5,ct(free)))
        #
        if add_C13:
            if not(np.allclose([θ%(2*np.pi),θ2%(2*np.pi),x_gate_angle%(2*np.pi)],np.zeros(3))):
                print('Warning: 2nd free evolution is calculated without the C13 coupling, because the effective model only works for both free evolutions when there are no MW gates between the of them')
            else:
                # this is to emulate the effective evolution induced by a nearby C-13 with parallel component γ13 (along z). 
                # For details see /home/santiago/Documents/Postdoc/2022_Postdoc_MIT/Roba di Mathematica/Hyperfine_NV_N_C13.nb 
                cosγC13 = 2*np.cos(γC13*tt) - 1/np.cos(γC13*tt) #np.cos(γC13*tt) # To effectively consider the time for the 1st free evolution we use 2*np.cos(γC13*tt) - 1/np.cos(γC13*tt)
                c13Eff = np.array([[      1,       1, cosγC13, cosγC13], 
                                   [      1,       1, cosγC13, cosγC13],
                                   [cosγC13, cosγC13,       1,       1],
                                   [cosγC13, cosγC13,       1,       1]])
                r6 = r6*c13Eff # note element-by-element product
        #
        if add_dephasing:
            if Γ_dephasing is None:
                r6 = r6*dephasing # note element-by-element product
            else:
                eΓt = np.exp(-Γ_dephasing*tt)
                dephasing_tt = np.array([[1,     1, eΓt, eΓt],
                                         [1,     1, eΓt, eΓt],
                                         [eΓt, eΓt,   1,   1],
                                         [eΓt, eΓt,   1,   1]])
                r6 = r6*dephasing_tt # note element-by-element product
        
        if do_spin_echo_0:
            r6 = mul(rxypi,    mul(r6,ct(rxypi)))
        #solx[i] = 0.5-0.5*np.trace(mul(σx,traceS(r6)))
        #soly[i] = 0.5-0.5*np.trace(mul(σy,traceS(r6)))

        #phi = (tt + t_extra/2/2)*A/(1)
        phi = -phi_factor*(tt*2 + t_extra)*A_phi/2 #A/2 #
        #phi = phi_factor*(tt)*2*A/2
        phi2= phi - phi_factor2*(t_extra2)*A_phi/2 #A/2 # add the last pi pulse
        if ideal_readout:
            #check sign 0.5-0.5 or 0.5+0.5
            solx[i] = 0.5-0.5*np.trace(mul(-np.cos(phi)*σy+np.sin(phi)*σx,traceS(r6))) 
            soly[i] = 0.5-0.5*np.trace(mul( np.cos(phi)*σx+np.sin(phi)*σy,traceS(r6)))
        else:
            # Instead of measuring np.cos(ϕ)*σx+np.sin(ϕ)*σy and np.cos(ϕ)*σy-np.sin(ϕ)*σx we apply:
            # nuclear(pi/2,conditional) + electronic(pi) + nuclear(pi/2,conditional)
            #n_rympi2_phi = expm(-1j*(np.pi/2)*fact_n*(1/Ω_n)*hamilt_H(0,δ_high,0, Ω_n,δ_n,-np.pi/2-phi, A))
            n_rympi2_phi =  expm(-1j*(n_half_pi_length)*hamilt_H(0,δ_high,0, Ω_n,δ_n,-(-np.pi/2+phi), A)) # MINUS PHASE
            #n_rxpi2_phi  = expm(-1j*(np.pi/2)*fact_n*(1/Ω_n)*hamilt_H(0,δ_high,0, Ω_n,δ_n,-phi        , A))
            n_rxpi2_phi  =  expm(-1j*(n_half_pi_length)*hamilt_H(0,δ_high,0, Ω_n,δ_n,-(+phi)        , A)) # MINUS PHASE
            #
            n_rympi2_phi2 =  expm(-1j*(n_half_pi_length)*hamilt_H(0,δ_high,0, Ω_n,δ_n,-(-np.pi/2+phi2), A)) # MINUS PHASE
            #n_rxpi2_phi  = expm(-1j*(np.pi/2)*fact_n*(1/Ω_n)*hamilt_H(0,δ_high,0, Ω_n,δ_n,-phi        , A))
            n_rxpi2_phi2  =  expm(-1j*(n_half_pi_length)*hamilt_H(0,δ_high,0, Ω_n,δ_n,-(+phi2)        , A)) # MINUS PHASE
            #
            r7_Re = mul(n_rympi2_phi, mul(r6   ,ct(n_rympi2_phi)))
            #r8_Re = mul(rxpi,    mul(r7_Re,ct(rxpi)))
            if do_spin_echo_1:
                r8_Re = mul(rxypi,    mul(r7_Re,ct(rxypi)))
            else:
                r8_Re = r7_Re
            if do_3rd_Rf:
                r9_Re = mul(n_rympi2_phi2, mul(r8_Re,ct(n_rympi2_phi2)))
            else:
                r9_Re = r8_Re
            if do_spin_echo_2:
                #r9_Re = mul(rxypi,    mul(r9_Re,ct(rxypi)))
                r9_Re = mul(rxypi_2,    mul(r9_Re,ct(rxypi_2)))
            if short_laser:
                # Here the correct sign is 0.5-0.5
                # also, can we improve how to simulate the effect of the laser?
                solx[i] = 0.5-0.5*np.trace(mul(σz,traceS(r9_Re))) 
            else:
                if do_CNOT:
                    r9_Re = mul(cnot_h,    mul(r9_Re,ct(cnot_h)))
                solx[i] = 0.5+0.5*np.trace(mul(σz,traceA(r9_Re)))
            #
            #
            r7_Im = mul(n_rxpi2_phi, mul(r6   ,ct(n_rxpi2_phi)))
            #r8_Im = mul(rxpi,    mul(r7_Im,ct(rxpi)))
            if do_spin_echo_1:
                r8_Im = mul(rxypi,    mul(r7_Im,ct(rxypi)))
            else:
                r8_Im = r7_Im
            if do_3rd_Rf:
                r9_Im = mul(n_rxpi2_phi2, mul(r8_Im,ct(n_rxpi2_phi2)))
            else:
                r9_Im = r8_Im
            if do_spin_echo_2:
                #r9_Im = mul(rxypi,    mul(r9_Im,ct(rxypi)))
                r9_Im = mul(rxypi_2,    mul(r9_Im,ct(rxypi_2)))
            if short_laser:
                # Here the correct sign is 0.5-0.5
                # also, can we improve how to simulate the effect of the laser?
                soly[i] = 0.5-0.5*np.trace(mul(σz,traceS(r9_Im)))
            else:
                if do_CNOT:
                    r9_Im = mul(cnot_h,    mul(r9_Im,ct(cnot_h)))
                soly[i] = 0.5+0.5*np.trace(mul(σz,traceA(r9_Im)))
            ##
            if return_S:
                # Note: returning rho before the laser/cnot
                rho_s_Re[i] = traceA(r9_Re)
                rho_s_Im[i] = traceA(r9_Im)
            ##
    if return_S:
        return rho_s_Re,rho_s_Im
    return solx,soly


##
## Ideal experiment 
## A=0 for all gates besides the free evolution and (and the phase of second pi/2 of the nuclear spin)
##
def ideal_experiment(u_vec,r0,θ,θ2, x_gate_angle=np.pi,pulse1_angle=np.pi/2,Ω_high=Ω_high,δ_high=δ_high,A=A,A_phi=A_phi,fact_n_ideal = 1,phi_factor=1):
    # Definition of nuclear gates
    n_rxmpi2 = expm(+1j*(np.pi/2)*fact_n_ideal*(1/Ω_n)*hamilt_H(0,δ_high,0, Ω_n,δ_n,np.pi/2, 0*A)) # R_x(-π/2)
    
    # Definition of gates
    rypi2_ideal = expm(-1j*(pulse1_angle)*(1/Ω_high)*hamilt_H(Ω_high,δ_high,np.pi/2, 0,δ_n,0, 0*A)) # R_y(π/2)
    ryθ_ideal   = expm(-1j*(θ)*(1/Ω_high)*hamilt_H(Ω_high,δ_high,np.pi/2, 0,δ_n,0, 0*A)) # R_y(θ)
    #rymθ_ideal = expm(+1j*(θ)*(1/Ω_high)*hamilt_H(Ω_high,δ_high,np.pi/2, 0,δ_n,0, 0*A)) # R_y(-θ)
    rymθ2_ideal = expm(+1j*(θ2)*(1/Ω_high)*hamilt_H(Ω_high,δ_high,np.pi/2, 0,δ_n,0, 0*A)) # R_y(-θ2)
    rxpi_ideal  = expm(-1j*(x_gate_angle)*(1/Ω_high)*hamilt_H(Ω_high,δ_high,0, 0,δ_n,0, 0*A)) # R_x(π)

    #print(np.allclose(u_vec,xdat0))
    solx_ideal = np.zeros(len(u_vec),dtype=complex)
    soly_ideal = np.zeros(len(u_vec),dtype=complex)

    # first pi/2 nuclear spin:
    r0 = mul(n_rxmpi2, mul(r0,ct(n_rxmpi2)))
    
    # first pi/2 e- spin:
    r1 = mul(rypi2_ideal, mul(r0,ct(rypi2_ideal)))

    for i,tt in enumerate(u_vec):
        free = expm(-1j*tt*hamilt_H(0,δ_high,0, 0,δ_n,0, A)) # free evolution

        r2 = mul(free, mul(r1,ct(free)))
        r3 = mul(ryθ_ideal,  mul(r2,ct(ryθ_ideal)))
        r4 = mul(rxpi_ideal, mul(r3,ct(rxpi_ideal)))
        #r5 = mul(rymθ_ideal, mul(r4,ct(rymθ_ideal)))
        r5 = mul(rymθ2_ideal, mul(r4,ct(rymθ2_ideal)))
        r6 = mul(free, mul(r5,ct(free)))

        #solx[i] = 0.5-0.5*np.trace(mul(σx,traceS(r6)))
        #soly[i] = 0.5-0.5*np.trace(mul(σy,traceS(r6)))

        phi = phi_factor*tt*A_phi/(1) #A/(1) #
        # Ideal readout:
        solx_ideal[i] = 0.5-0.5*np.trace(mul(np.cos(phi)*σx+np.sin(phi)*σy,traceS(r6)))
        soly_ideal[i] = 0.5-0.5*np.trace(mul(np.cos(phi)*σy-np.sin(phi)*σx,traceS(r6)))
        #soly_ideal[i] = 0.5-0.5*np.trace(mul(-np.cos(phi)*σy+np.sin(phi)*σx,traceS(r6)))
    return solx_ideal,soly_ideal


