###
### Program to calculate the dynamics of a 3-level sytem in the presence of
### POVM and dissipation (towards |0>) for two different Hamiltonians: H_mw, H_NV
### In the $S_Z$ basis, the Hamiltonians (in rotating frame) are:
### $\mathcal{H}_\mathrm{mw} = (1/\sqrt(2))\begin{pmatrix} 0 & \Omega & 0 \\ \Omega & 0 & \Omega \\ 0 & \Omega & 0 \end{pmatrix}$
### $\mathcal{H}_\mathrm{NV} = \begin{pmatrix} E_{+1} & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & E_{-1} \end{pmatrix}$
###



import numpy as np


# Dimensions of the system
dim_s = 3

# S_z eigenstates
rho_z = [np.array([[0,0,0], [0,0,0], [0,0,1]]), # ms=-1
         np.array([[0,0,0], [0,1,0], [0,0,0]]), # ms=0
         np.array([[1,0,0], [0,0,0], [0,0,0]])] # ms=+1
## Hamiltonian H_mw eigenstates (ordered in energy)
rho_Hmw = [0.25*np.array([[1,-np.sqrt(2),1], [-np.sqrt(2),2,-np.sqrt(2)], [1,-np.sqrt(2),1]]), # down
           0.5*np.array([[1,0,-1], [0,0,0], [-1,0,1]]),                                        # left
           0.25*np.array([[1, np.sqrt(2),1], [ np.sqrt(2),2, np.sqrt(2)], [1, np.sqrt(2),1]])] # up

## dissipation towards |0>
def dissip_propagator(t,ΓQ):
    """Propagator for the (dissipative) Lindbladian superoperator taking the system into the ms=0 state"""
    #fdy = np.array([[np.exp(-t*ΓQ),0,0,0],
    #                [0,np.exp(-t*ΓQ/2), 0, 0],
    #                [0, 0,np.exp(-t*ΓQ/2), 0],
    #                [1-np.exp(-t*ΓQ), 0, 0, 1]])
    fdy = np.array([[ np.exp(-t*ΓQ) , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ],
                    [ 0 , np.exp(-t*ΓQ/2) , 0 , 0 , 0 , 0 , 0 , 0 , 0 ],
                    [ 0 , 0 , np.exp(-t*ΓQ) , 0 , 0 , 0 , 0 , 0 , 0 ],
                    [ 0 , 0 , 0 , np.exp(-t*ΓQ/2) , 0 , 0 , 0 , 0 , 0 ],
                    [ 1-np.exp(-t*ΓQ) , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 1-np.exp(-t*ΓQ) ],
                    [ 0 , 0 , 0 , 0 , 0 , np.exp(-t*ΓQ/2) , 0 , 0 , 0 ],
                    [ 0 , 0 , 0 , 0 , 0 , 0 , np.exp(-t*ΓQ) , 0 , 0 ],
                    [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , np.exp(-t*ΓQ/2) , 0 ],
                    [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , np.exp(-t*ΓQ) ]])
    return fdy

##
## Propagator for H=H_mw
##
def unitary_propagator(t,ω):
    """Propagator for the unitary evolution superoperator for
    H = \frac{1}{\sqrt{2}}\begin{pmatrix}
    0 & ω & 0 \\
    ω & 0 & ω \\
    0 & ω & 0
    \end{pmatrix}"""
    # (from Dropbox/Experimental Jarzynski on Nv-center/Mathematica_codes/3Lvl-simple-model.nb)
    s1 = np.sin(t*ω)
    s2 = np.sin(2*t*ω)
    s05 = np.sin(t*ω/2)
    c1 = np.cos(t*ω)
    c05 = np.cos(t*ω/2)
    fdy=np.array(
        [[c05**4,(1j*(2*s1+s2))/(4*np.sqrt(2)),-(1/4)*s1**2,-((1j*(2*s1+s2))/(4*np.sqrt(2))),1/2*s1**2,(1j*s05**2*s1)/np.sqrt(2),-(1/4)*s1**2,-((1j*s05**2*s1)/np.sqrt(2)),s05**4],
        [(1j*(2*s1+s2))/(4*np.sqrt(2)),c05**2*c1,(1j*(2*s1+s2))/(4*np.sqrt(2)),1/2*s1**2,-((1j*s2)/(2*np.sqrt(2))),1/2*s1**2,-((1j*s05**2*s1)/np.sqrt(2)),-c1*s05**2,-((1j*s05**2*s1)/np.sqrt(2))],
        [-(1/4)*s1**2,(1j*(2*s1+s2))/(4*np.sqrt(2)),c05**4,(1j*s05**2*s1)/np.sqrt(2),1/2*s1**2,-((1j*(2*s1+s2))/(4*np.sqrt(2))),s05**4,-((1j*s05**2*s1)/np.sqrt(2)),-(1/4)*s1**2],
        [-((1j*(2*s1+s2))/(4*np.sqrt(2))),1/2*s1**2,(1j*s05**2*s1)/np.sqrt(2),c05**2*c1,(1j*s2)/(2*np.sqrt(2)),-c1*s05**2,-((1j*(2*s1+s2))/(4*np.sqrt(2))),1/2*s1**2,(1j*s05**2*s1)/np.sqrt(2)],
        [1/2*s1**2,-((1j*s2)/(2*np.sqrt(2))),1/2*s1**2,(1j*s2)/(2*np.sqrt(2)),c1**2,(1j*s2)/(2*np.sqrt(2)),1/2*s1**2,-((1j*s2)/(2*np.sqrt(2))),1/2*s1**2],
        [(1j*s05**2*s1)/np.sqrt(2),1/2*s1**2,-((1j*(2*s1+s2))/(4*np.sqrt(2))),-c1*s05**2,(1j*s2)/(2*np.sqrt(2)),c05**2*c1,(1j*s05**2*s1)/np.sqrt(2),1/2*s1**2,-((1j*(2*s1+s2))/(4*np.sqrt(2)))],
        [-(1/4)*s1**2,-((1j*s05**2*s1)/np.sqrt(2)),s05**4,-((1j*(2*s1+s2))/(4*np.sqrt(2))),1/2*s1**2,(1j*s05**2*s1)/np.sqrt(2),c05**4,(1j*(2*s1+s2))/(4*np.sqrt(2)),-(1/4)*s1**2],
        [-((1j*s05**2*s1)/np.sqrt(2)),-c1*s05**2,-((1j*s05**2*s1)/np.sqrt(2)),1/2*s1**2,-((1j*s2)/(2*np.sqrt(2))),1/2*s1**2,(1j*(2*s1+s2))/(4*np.sqrt(2)),c05**2*c1,(1j*(2*s1+s2))/(4*np.sqrt(2))],
        [s05**4,-((1j*s05**2*s1)/np.sqrt(2)),-(1/4)*s1**2,(1j*s05**2*s1)/np.sqrt(2),1/2*s1**2,-((1j*(2*s1+s2))/(4*np.sqrt(2))),-(1/4)*s1**2,(1j*(2*s1+s2))/(4*np.sqrt(2)),c05**4]]
    )
    return fdy

##
## Function to calculate the the superoperator B^n
##
def supOp_B_nL(n_L,b_parameters):
    """
    Function to calculate the superoperator (B)^{n_L}, with B=AU=(sum_j D_j m_j)U

    n_L = Number of laser pulses
    if len(parameters)==5: #(case H=H_mw)
        b_parameters = [pabs,ω,t_u,ΓQ,t_d] = [temperature factor,absorption probability,Rabi frequency in MHz,
                                               time of unitary evolution, dissipation rate,time of dissipation]
    elif len(parameters)==3: #(case H=H_NV)
        b_parameters = [pabs,ΓQ,t_d] = [temperature_factor X E_plus,temperature_factor X E_plus,
                                              absorption probability, dissipation rate,time of dissipation]
    returns 9x9 matrix (superoperator B)
    """
    ##
    ## Definition of constants and operators
    ##
    pabs = b_parameters[0]
    ΓQ   = b_parameters[-2]
    t_d  = b_parameters[-1]
    if len(b_parameters)==5: #(case H=H_mw)
        ω  = b_parameters[1]
        t_u= b_parameters[2]
        # Propagator for unitary evolution superoperator
        U_P = unitary_propagator(t_u,ω)
    elif len(b_parameters)==3: #(case H=H_NV)
        # Propagator for unitary evolution superoperator: H=∆S_z^2+γ_eBS_z ==> U gives global phase to S_z eigenstates
        U_P = np.eye(9,dtype=complex) # Since we don't care about phase (because of the QPMs), U=Identity
    ##
    ## Propagator for (dissipative) Lindbladian superoperator
    D_P = dissip_propagator(t_d,ΓQ)
    # selective dynamics depending on POVM result
    S_P = [D_P,D_P,D_P,np.eye(9)]
    #
    # POVM's operators
    pi_j = [pabs*proj for proj in rho_z] + [(1-pabs)*np.eye(3)]
    # POVM's measurement operators
    m_j_list=[np.sqrt(proj) for proj in pi_j]
    # POVM's superoperators
    π_j  = [np.kron(m_j,m_j) for m_j in m_j_list]
    # POVM+dissipation superoperators
    B_j = [np.matmul(S_P[j], np.matmul(π_j[j], U_P)) for j in range(len(π_j))]
    ## Propagator B = A.U (from left to right: unitary evol, POVM, dissipation)
    B_P = np.round(sum(B_j),20) #rounded to 20 decimals to simplify computation of matrix_power
    # Power of B_P for n_L laser pulses
    B_P_nL = np.linalg.matrix_power(B_P,n_L)
    return B_P_nL

##
## Function to calculate the LHS after n_L laser pulses
##
def calc_lhs_n(n_L, parameters):
    """
    Function to calculate the LHS after n_L laser pulses

    n_L = Number of laser pulses
    if len(parameters)==6: #(case H=H_mw)
        parameters = [βω,pabs,ω,t_u,ΓQ,t_d] = [temperature factor,absorption probability,Rabi frequency in MHz,
                                               time of unitary evolution, dissipation rate,time of dissipation]
    elif len(parameters)==5: #(case H=H_NV)
        parameters = [βEp,βEm,pabs,ΓQ,t_d] = [temperature_factor X E_plus,temperature_factor X E_minus,
                                              absorption probability, dissipation rate,time of dissipation]

    returns lhs(n_L)
    """
    ##
    ## Definition of constants and operators
    ##
    if len(parameters)==6: #(case H=H_mw)
        βω,*b_parameters = parameters # => len(b_parameters)=5
        ## Hamiltonian eigenstates (ordered in energy)
        rho_H = rho_Hmw
        # Initial thermal state
        ρ_th = (rho_H[2]*np.exp(-βω) + rho_H[1] + rho_H[0]*np.exp(βω))/(np.exp(βω)+1+np.exp(-βω))
        # Expectation values of <e^(-β H)>
        expβE = [np.exp(+βω),1,np.exp(-βω)] # [down, left, up] respectively
    elif len(parameters)==5: #(case H=H_NV)
        βEp,βEm,*b_parameters = parameters # => len(b_parameters)=3
        ## Hamiltonian eigenstates (ordered in energy)
        rho_H = rho_z
        # Initial thermal state
        ρ_th = (rho_H[2]*np.exp(-βEp) + rho_H[1] + rho_H[0]*np.exp(-βEm))/(np.exp(-βEp)+1+np.exp(-βEm))
        # Expectation values of <e^(-β H)>
        expβE = [np.exp(-βEm),1,np.exp(-βEp)] # [-1, 0, +1] respectively
    ##
    ##
    # Power B^{n_L}, where B = A.U (from left to right: unitary evol, POVM, dissipation)
    B_P_nL = supOp_B_nL(n_L,b_parameters)

    # Super - projectors on the Hamiltonian basis
    p_H  = [np.kron(rho,rho) for rho in rho_H]
    # Vectorialized thermal state
    ρ_th_v = ρ_th.reshape([dim_s*dim_s])

    ##
    ## Calculate the probability P(k,i), and P(k,i)exp(-\beta(E_k-E_i))
    ##
    pki_list = []
    lhs_aux = []
    for i in range(dim_s):
        for k in range(dim_s):
            pki=(np.matmul(p_H[k], np.matmul(B_P_nL, np.matmul(p_H[i], ρ_th_v)))).reshape(3,3).trace()
            pki_list.append(pki)
            lhs_aux.append( pki*expβE[k]/expβE[i] )
    result = sum(lhs_aux)
    if not(np.isclose(result.imag,0)):
        print('Error! Imaginary characteristic function G(\beta)!')
    return result.real


##
## Function to calculate the parameter gamma after n_L laser pulses
##
def calc_γ_n(n_L,parameters):
    """
    n_L = Number of laser pulses
    if len(parameters)==6: #(case H=H_mw)
        parameters = [βω,pabs,ω,t_u,ΓQ,t_d] = [temperature factor,absorption probability,Rabi frequency in MHz,
                                               time of unitary evolution, dissipation rate,time of dissipation]
    elif len(parameters)==5: #(case H=H_NV)
        parameters = [βEp,βEm,pabs,ΓQ,t_d] = [temperature_factor X E_plus,temperature_factor X E_minus,
                                              absorption probability, dissipation rate,time of dissipation]
    Returns: parameter gamma after n_L laser pulses
    """
    ##
    ## Definition of constants and operators
    ##
    if len(parameters)==6: #(case H=H_mw)
        βω,*b_parameters = parameters # => len(b_parameters)=5
        ## Hamiltonian eigenstates (ordered in energy)
        rho_H = rho_Hmw
        # Thermal state
        ρ_th = (rho_H[2]*np.exp(-βω) + rho_H[1] + rho_H[0]*np.exp(βω))/(np.exp(βω)+1+np.exp(-βω))
    elif len(parameters)==5: #(case H=H_NV)
        βEp,βEm,*b_parameters = parameters # => len(b_parameters)=3
        ## Hamiltonian eigenstates (ordered in energy)
        rho_H = rho_z
        # Thermal state
        ρ_th = (rho_H[2]*np.exp(-βEp) + rho_H[1] + rho_H[0]*np.exp(-βEm))/(np.exp(-βEp)+1+np.exp(-βEm))
    ##
    ##
    # Power B^{n_L}, where B = A.U (from left to right: unitary evol, POVM, dissipation)
    B_P_nL = supOp_B_nL(n_L,b_parameters)
    # Vectorialized thermal state
    ρ_th_v = ρ_th.reshape([dim_s*dim_s])

    # Re-shape the product: mat_fin[3x3] = col^{-1}( (B^{n_L})^\dagger col[\rho^{th}] )
    mat_fin = np.reshape( np.matmul(B_P_nL.conjugate().transpose() , ρ_th_v) , [3,3])
    # calculate \gamma = Tr[ mat_fin ]:
    result = mat_fin.trace()
    if not(np.isclose(result.imag,0)):
        print('Error! Imaginary trace!')
    return result.real



def conditional_probs(n_L, b_parameters):
    """
    Function to calculate energy jump conditional probabilities after n_L laser pulses

    n_L = Number of laser pulses
    if len(parameters)==5: #(case H=H_mw)
        b_parameters = [pabs,ω,t_u,ΓQ,t_d] = [temperature factor,absorption probability,Rabi frequency in MHz,
                                               time of unitary evolution, dissipation rate,time of dissipation]
    elif len(parameters)==3: #(case H=H_NV)
        b_parameters = [pabs,ΓQ,t_d] = [temperature_factor X E_plus,temperature_factor X E_plus,
                                              absorption probability, dissipation rate,time of dissipation]
    returns list of conditional probs [p00,p10,p20, p01,p11,p21, p02,p12,p22]
    """
    ##
    # Power B^{n_L}, where B = A.U (from left to right: unitary evol, POVM, dissipation)
    B_P_nL = supOp_B_nL(n_L,b_parameters)
    #
    if len(b_parameters)==5: #(case H=H_mw)
        ## Hamiltonian eigenstates (ordered in energy)
        rho_H = rho_Hmw
    elif len(b_parameters)==3: #(case H=H_NV)
        ## Hamiltonian eigenstates (ordered in energy)
        rho_H = rho_z
    # Super - projectors on the Hamiltonian basis
    p_H  = [np.kron(rho,rho) for rho in rho_H]
    # Vectorialized Hamiltonian eigenstates (ordered in energy)
    rho_H_v_list = [rho.reshape([dim_s*dim_s]) for rho in rho_H]
    ##
    ## Calculate the probability P(k|i)
    ##
    Pki_list = []
    for i in range(dim_s):
        for k in range(dim_s):
            Pki=(np.matmul(p_H[k], np.matmul(B_P_nL, rho_H_v_list[i]))).reshape(3,3).trace()
            Pki_list.append(Pki)
    result = np.array(Pki_list)
    if not(np.allclose(result.imag,0)):
        print('Error! Imaginary probabilities!')
    return result.real

def conditional_probs_EPM(n_L, b_parameters, rho_0_vec):
    """
    Function to calculate energy jump conditional probabilities after n_L laser pulses starting from a given state

    n_L = Number of laser pulses
    if len(parameters)==5: #(case H=H_mw)
        b_parameters = [pabs,ω,t_u,ΓQ,t_d] = [temperature factor,absorption probability,Rabi frequency in MHz,
                                               time of unitary evolution, dissipation rate,time of dissipation]
    elif len(parameters)==3: #(case H=H_NV)
        b_parameters = [pabs,ΓQ,t_d] = [temperature_factor X E_plus,temperature_factor X E_plus,
                                              absorption probability, dissipation rate,time of dissipation]
    returns list of conditional probs [p00,p10,p20, p01,p11,p21, p02,p12,p22]
    """
    ##
    # Power B^{n_L}, where B = A.U (from left to right: unitary evol, POVM, dissipation)
    B_P_nL = supOp_B_nL(n_L,b_parameters)
    #
    if len(b_parameters)==5: #(case H=H_mw)
        ## Hamiltonian eigenstates (ordered in energy)
        rho_H = rho_Hmw
    elif len(b_parameters)==3: #(case H=H_NV)
        ## Hamiltonian eigenstates (ordered in energy)
        rho_H = rho_z
    # Super - projectors on the Hamiltonian basis
    p_H  = [np.kron(rho,rho) for rho in rho_H]
    ##
    ## Calculate the probability P(k|i)
    ##
    Pki_list = []
    for k in range(dim_s):
        Pki=(np.matmul(p_H[k], np.matmul(B_P_nL, rho_0_vec))).reshape(3,3).trace()
        Pki_list.append(Pki)
    result = np.array(Pki_list)
    if not(np.allclose(result.imag,0)):
        print('Error! Imaginary probabilities!')
    return result.real


##
## Function to calculate the Shannon entropy of the trajectories
##
def shS(n_L,parameters,just_return_pk_list=False):
    """
    Function to calculate the Shannon entropy of all trajectories

    n_L number of laser pulses
    if len(parameters)==6: #(case H=H_mw)
        parameters = [βω,pabs,ω,t_u,ΓQ,t_d] = [temperature factor,absorption probability,Rabi frequency in MHz,
                                               time of unitary evolution, dissipation rate,time of dissipation]
    elif len(parameters)==5: #(case H=H_NV)
        parameters = [βEp,βEm,pabs,ΓQ,t_d] = [temperature_factor X E_plus,temperature_factor X E_minus,
                                              absorption probability, dissipation rate,time of dissipation]

    returns shannon entropy [float]
    """
    ##
    ## Definition of constants and operators
    ##
    if len(parameters)==6: #(case H=H_mw)
        βω,*b_parameters = parameters # => len(b_parameters)=5
        ## Hamiltonian eigenstates (ordered in energy)
        rho_H = rho_Hmw
        # Initial thermal state
        ρ_th = (rho_H[2]*np.exp(-βω) + rho_H[1] + rho_H[0]*np.exp(βω))/(np.exp(βω)+1+np.exp(-βω))
        # Expectation values of <e^(-β H)>
        expβE = [np.exp(+βω),1,np.exp(-βω)] # [down, left, up] respectively
        #
        ω  = b_parameters[1]
        t_u= b_parameters[2]
        # Propagator for unitary evolution superoperator
        U_P = unitary_propagator(t_u,ω)
    elif len(parameters)==5: #(case H=H_NV)
        βEp,βEm,*b_parameters = parameters # => len(b_parameters)=3
        ## Hamiltonian eigenstates (ordered in energy)
        rho_H = rho_z
        # Initial thermal state
        ρ_th = (rho_H[2]*np.exp(-βEp) + rho_H[1] + rho_H[0]*np.exp(-βEm))/(np.exp(-βEp)+1+np.exp(-βEm))
        # Expectation values of <e^(-β H)>
        expβE = [np.exp(-βEm),1,np.exp(-βEp)] # [-1, 0, +1] respectively
        #
        # Propagator for unitary evolution superoperator: H=∆S_z^2+γ_eBS_z ==> U gives global phase to S_z eigenstates
        U_P = np.eye(9,dtype=complex) # Since we don't care about phase (because of the QPMs), U=Identity
    pabs = b_parameters[0]
    ΓQ   = b_parameters[-2]
    t_d  = b_parameters[-1]
    ##
    ## Propagator for (dissipative) Lindbladian superoperator
    D_P = dissip_propagator(t_d,ΓQ)
    # selective dynamics depending on POVM result
    S_P = [D_P,D_P,D_P,np.eye(9)]
    #
    # POVM's operators
    pi_j = [pabs*proj for proj in rho_z] + [(1-pabs)*np.eye(3)]
    # POVM's measurement operators
    m_j_list=[np.sqrt(proj) for proj in pi_j]
    # POVM's superoperators
    π_j  = [np.kron(m_j,m_j) for m_j in m_j_list]
    # POVM+dissipation superoperators
    B_j = [np.matmul(S_P[j], np.matmul(π_j[j], U_P)) for j in range(len(π_j))]
    # Vectorialized thermal state
    ρ_th_v = ρ_th.reshape([dim_s*dim_s])
    #
    ## Definition of auxiliary functions
    def BπU(rho_v,j):
        return np.matmul(B_j[j], rho_v)
    def loop_rec_BπU(y, n):
        if n<0:
            print('negative recursive variable')
            return -1
        elif n >= 1:
            y_aux = []
            for j_n in range(len(π_j)):
                y_aux = y_aux + [BπU(y_nm1,j_n) for y_nm1 in y]
            y = y_aux
            #loop_rec_BπU(y, n - 1)
            return loop_rec_BπU(y, n - 1)
        return y
    ##
    ## Calculation of single trajectory probabilities
    ##
    loop_result_list = loop_rec_BπU([ρ_th_v],n_L)
    print('Number of trajectories =',len(loop_result_list))
    probs_k_list_n = np.zeros(len(loop_result_list))
    for ii in range(len(probs_k_list_n)):
        loop_result = loop_result_list[ii]
        probs_k_list_n[ii] = loop_result.reshape(3,3).trace().real
    if just_return_pk_list:
        return probs_k_list_n
    return np.nansum(probs_k_list_n*np.log(probs_k_list_n))
