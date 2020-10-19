###
### Program to calculate the dynamics of a 3-level sytem in the presence of z-QPMs and dissipation (towards |0>)
### In the $S_Z$ basis, the Hamiltonian (in rotating frame) is:
### $\mathcal{H} = \begin{pmatrix} -\delta_+ & \Omega_+ & 0 \\ \Omega_+ & 0 & \Omega_- \\ 0 & \Omega_- & -\delta_- \end{pmatrix}$
###

# About optimization on the matrix exponential calculation:
#
# From DOI: 10.2197/ipsjjip.23.171
# "Let $S$ be an element of Sym(3). Then, the matrix exponential of $S$ is $exp(S)=P exp(D) ^tP$, where $P$ is an orthogonal matrix and $D$ is a diagonal matrix. $P$ and $D$ satisfy $S=PD^tP$".
#
# From https://www.math.ubc.ca/~pwalls/math-python/linear-algebra/eigenvalues-eigenvectors/
# "A beautiful result in linear algebra is that a square matrix $M$ of size $n$ is diagonalizable if and only if $M$ has $n$ independent eigevectors. Furthermore, $M=PD P^{-1}$ where the columns of $P$ are the eigenvectors of $M$ and $D$ has corresponding eigenvalues along the diagonal".


import numpy as np
from scipy.linalg import eig

# Useful matrices:
Sz = np.array([[1,0,0],[0,0,0],[0,0,-1]])
# Z projectors
proj1 = np.array([[1,0,0],[0,0,0],[0,0,0]])
proj0 = np.array([[0,0,0],[0,1,0],[0,0,0]])
projm1 = np.array([[0,0,0],[0,0,0],[0,0,1]])


# Function to propagate rho0 in time under H.
def calcRho(rho0,times,Ωp,δp,Ωm,δm,digitsPrecision=8):
    """
    Parameters:
    rho0 = initial density matrix in the S_z basis (order: ms = +1,0,-1)
    times = time vector to calculate the dynamics
    Ωp = |0> <--> |+1> Rabi freq.
    δp = |0> <--> |+1> detuning
    Ωm = |0> <--> |-1> Rabi freq.
    δm = |0> <--> |-1> detuning
    Optional parameters:
    digitsPrecision = number of digits to round up calculation

    Return:
    rho_t = Array of shape (n,3,3) where n is the length of vector times, and for each n the
            3x3 matrix represents the state for the time t=times[n]
    """

    # Hamiltonian in the S_z basis (order: ms = +1,0,-1)
    #H = np.array([[-δp,Ωp,0],[Ωp,0,Ωm],[0,Ωm,-δm]])
    H = np.array([[-δp,(Ωp+0*1j).conjugate(),0],[Ωp,0,(Ωm+0*1j).conjugate()],[0,Ωm,-δm]])

    # Hamiltonian eigenvalues and eigenstates (used to build the P and D matrices)
    eVals,eVecs = eig(H)
    eVals,eVecs = np.round(eVals,8),np.round(eVecs,8)
    # Note: The P matrix is P=eVecs

    # Diagonal matrix with the eigen-values along the diagonal
    D = np.array([[eVals[0],0,0],[0,eVals[1],0],[0,0,eVals[2]]])

    # The following statements are True (just a test of the definitions)
    #print('H = P D (P)^† →',np.allclose( H , np.dot(eVecs,np.dot(D,eVecs.transpose())) ) )
    #print('D = (P)^† H P →',np.allclose( D , np.dot(eVecs.transpose(),np.dot(H,eVecs)) ) )

    # Array to store rho:
    rho_t = np.zeros([len(times),3,3],dtype=complex)

    # Iteration over the time vector
    for i,t in enumerate(times):
        # The exponential of a diagonal matrix is the exponential of the diagonal elements:
        expD = np.array([[np.exp(-1j*t*eVals[0]),0,0],[0,np.exp(-1j*t*eVals[1]),0],[0,0,np.exp(-1j*t*eVals[2])]])
        # Unitary evoultion operator: U(t) = exp(-iHt) = P.exp(-iDt).(P)^†
        opU = np.dot(eVecs,np.dot(expD,eVecs.transpose().conjugate()))
        # state for a time t: rho(t) = U(t).rho0.(U(t))^†
        rho_t[i] = np.dot(opU,np.dot(rho0,opU.conj().transpose()))
    # Remove small numbers (errors during calculation)
    rho_t = np.round(rho_t,digitsPrecision)
    return rho_t


# Function to simulate a z-QPM, doing 2 coin flips
def coin_flip_3lvls(rho, beta=None, g_1m=79.7, gamma=77):
    """
    Parameters:
    rho0 = density matrix in the S_z basis (order: ms = +1,0,-1)
    beta = non-spin conserving probability
    g_1m = (non-radiative) decay rate from excited |+-1> state to metastable state
    gamma = radiative decay rate from any excited state to its correspondent ground state
    Return:
    projk = where k is +1, 0, or -1, depending on the result of the coin flips
    """
    # Probability of projecting into ground state
    #prob_proj1 = rho[0,0] #np.trace(proj1*rho) #
    prob_projm1 = rho[2,2] #np.trace(projm1*rho) #
    prob_proj0 = rho[1,1] #1 - (prob_proj1 + prob_projm1) #
    #
    P0ex = prob_proj0 + beta*(1 - 3*prob_proj0) # Probability that, starting from rho, I excite in |0>
    #P1ex = prob_proj1 + beta*(prob_proj0 - prob_proj1) # Probability that, starting from rho, I excite in |1>
    Pm1ex = prob_projm1 + beta*(prob_proj0 - prob_projm1) # Probability that, starting from rho, I excite in |-1>
    #
    #P0dec = P0ex*(1-3*beta) + beta
    #P1dec = P1ex + beta*(P0ex - P1ex)
    #Pm1dec = Pm1ex + beta*(P0ex - Pm1ex)
    #
    #Pm1_coin = Pm1dec
    #P0_coin = Pm1dec + P0dec
    #

    if beta is None:
        beta = 0.01

    Pm1_coin = Pm1ex + beta*(P0ex - Pm1ex)
    P0_coin = Pm1_coin + P0ex*(1-3*beta) + beta
    #
    rand_number,rand_number2 = np.random.rand(2) # for QMP: where we project and for re-init
    if rand_number2 <= (g_1m/(g_1m+gamma)): # re-init? Should be the same with 2 or 3 levels
        return proj0 # Dissipation in mS = 0
    else:    # Radiative decay
        if rand_number <= Pm1_coin: # project into ms=-1
            return projm1 #ms = -1
        elif rand_number <= P0_coin:# and rand_number > Pm1_coin: # project into ms=0
            return proj0  #ms = 0
        else: #Project into mS = +1
            return proj1  # ms = 1


# Calculate the eigenvectors of H
def changeBasisMat(Ωp,δp,Ωm,δm,digitsPrecision=8):
    """
    Parameters:
    Ωp = |0> <--> |+1> Rabi freq.
    δp = |0> <--> |+1> detuning
    Ωm = |0> <--> |-1> Rabi freq.
    δm = |0> <--> |-1> detuning
    Optional parameters:
    digitsPrecision = number of digits to round up calculation
    Return:
    eVecs = Array of shape (3,3) where each row is an eigenvector of H
    """

    # Hamiltonian in the S_z basis (order: ms = +1,0,-1)
    #H = np.array([[-δp,Ωp,0],[Ωp,0,Ωm],[0,Ωm,-δm]])
    H = np.array([[-δp,(Ωp+0*1j).conjugate(),0],[Ωp,0,(Ωm+0*1j).conjugate()],[0,Ωm,-δm]])

    # Hamiltonian eigenvalues and eigenstates (used to build the P and D matrices)
    eVals,eVecs = eig(H)
    eVals,eVecs = np.round(eVals,digitsPrecision),np.round(eVecs,digitsPrecision)
    return eVecs


# Function to simmulate the dynamics under z-QPMs with dissipation
def threeLvls_DissDyn(rho0,time_unitary,timeStep = 1,nL = 10,prob_abs = 0.25,
                      ν = 1.18e-3,δp_factor=0,δm_factor=0,ϕ = 0, beta=None, digitsPrecision=8):
    """
    rho0 = Initial state
    time_unitary = time between laser pulses -- unitary evolution (in ns)

    Optional parameters:
    timeStep = time step (in ns)
    nL = number of laser pulses
    prob_abs = absorption probability
    ν = Rabi frequency (in GHz)
    δp_factor = factor do define the detuning δp = δp_factor*Ωp
    δm_factor = factor do define the detuning δm = δm_factor*Ωm
    ϕ = phase of the Hamiltonian (ϕ=0 --> X, ϕ=π/2 --> Y)
    digitsPrecision = number of digits to round up calculation

    Return:
    rhoVector
    """

    flags_abs = prob_abs > np.random.rand(nL) # bool array: True --> absorb laser, False --> no absorption
    num_eff_abs = flags_abs.sum() # Number of absorbed lasers
    eff_abs_pos = np.where(flags_abs)[0] # Indices of the absorbed lasers
    # For the case of not even one absorption:
    if num_eff_abs==0:
        eff_abs_pos = np.array([nL-1])
        num_eff_abs=1
        # we impose one absorption at the very end of the protocol

    # Hamiltonian parameters:
    Ωp = 2*np.pi*ν # 2π.GHz
    δp = δp_factor*Ωp # 2π.GHz
    Ωm = 2*np.pi*ν # 2π.GHz
    δm = δm_factor*Ωm # 2π.GHz

    #secondTimeVec = np.array([0]) #for debugging

    # Array to store rho (in the z basis)
    rho_size = int(1 + nL*time_unitary/timeStep)
    rhoVector = np.zeros([rho_size,3,3],dtype=complex)
    # First element of vector
    rhoVector[0]=rho0
    # index to assign elements in rhoVector
    index_rho=0

    # Iteration over the periods before and after the absorbed laser pulses. The iteration finishes
    # at (num_eff_abs + 0) if last absorption is the last laser pulse, or
    # at (num_eff_abs + 1) if last absorption is not the last laser pulse.
    for i_abs in range(num_eff_abs + int(eff_abs_pos[-1]!=(nL-1))):
        if i_abs==0: # for the first iteration
            timeFin_unitary = time_unitary*(eff_abs_pos[i_abs] + 1)
        elif i_abs==(num_eff_abs): # for period before last laser but after last absorption
            timeFin_unitary = time_unitary*(nL-eff_abs_pos[i_abs-1]-1)
        else: # for generic iteration
            timeFin_unitary = time_unitary*(eff_abs_pos[i_abs]-eff_abs_pos[i_abs-1])

        #
        # Time of the unitary evolution (in ns)
        #timeVec_unitary = np.arange(0,np.nextafter(timeFin_unitary,timeFin_unitary-1) + timeStep,timeStep)
        timeVec_unitary = np.arange(0,timeFin_unitary + timeStep,timeStep)
        ## NOTE: instead of writing "timeFin_unitary", i wrote "np.nextafter(timeFin_unitary,timeFin_unitary-1)"
        ## which is the previous float value to "timeFin_unitary". This was necessary because for some particular
        ## time values the function np.arange was rounding up some precision digits, resulting in a longer array

        # Propagate unitary dyn. for H = [-δp,(Ωp.exp(ϕ))^*,0],[Ωp.exp(ϕ),0,(Ωm.exp(ϕ))^*],[0,Ωm.exp(ϕ),-δm]
        rho_t = calcRho(rho0,timeVec_unitary,Ωp*np.exp(ϕ*1j),δp,Ωm*np.exp(ϕ*1j),δm,digitsPrecision=digitsPrecision)

        prev_index_rho = 1*index_rho
        index_rho+=int(timeFin_unitary/timeStep)
        #print(prev_index_rho,index_rho)
        rhoVector[prev_index_rho+1:index_rho+1] = rho_t[1:]
        #secondTimeVec=np.concatenate([secondTimeVec,timeVector[prev_index_rho+1:index_rho+1]] ) #for debugging
        #secondTimeVec=np.concatenate([secondTimeVec,secondTimeVec[-1]+timeVec_unitary[1:]] ) #for debugging

        #rho0 = rho_t[-1]
        rho0 = coin_flip_3lvls(rho_t[-1], beta=beta)

    return rhoVector


# Function to calcualte the mean density matrix under the dissipative dynamics
def mean_rho_DissDyn(n_rep,init_eigenstate,time_unitary,timeStep = 1,nL = 10,prob_abs = 0.25,
                      ν = 1.18e-3,δp_factor=0,δm_factor=0,ϕ = 0, beta=0.01,
                      change_basis =True, digitsPrecision =8):
    """
    n_rep = number of repetitions
    init_eigenstate = Initial eigenstate: 'up', 'down' or 'left'
    time_unitary = time between laser pulses -- unitary evolution (in ns)

    Optional parameters:
    timeStep = time step (in ns)
    nL = number of laser pulses
    prob_abs = absorption probability
    ν = Rabi frequency (in GHz)
    δp_factor = factor do define the detuning δp = δp_factor*Ωp
    δm_factor = factor do define the detuning δm = δm_factor*Ωm
    ϕ = phase of the Hamiltonian (ϕ=0 --> X, ϕ=π/2 --> Y)
    change_basis = Flag to change basis
    digitsPrecision = number of digits to round up calculation

    Returns
    timeVector,rho_mean = time vector (in us), array of with mean rho for each time
    """

    # Hamiltonian parameters:
    Ωp = 2*np.pi*ν # 2π.GHz
    δp = δp_factor*Ωp # 2π.GHz
    Ωm = 2*np.pi*ν # 2π.GHz
    δm = δm_factor*Ωm # 2π.GHz

    # Matrix with the eigenvectors of H (to be used for the change of basis)
    p_mat = changeBasisMat(Ωp*np.exp(ϕ*1j),δp,Ωm*np.exp(ϕ*1j),δm,digitsPrecision=digitsPrecision)

    # Eigenvectors of H
    rho_up,rho_left,rho_down = [np.kron(eVec0,eVec0.conjugate()).reshape([3,3]) for eVec0 in p_mat.transpose()]

    # time of the experiment in ns
    timeVector = np.arange(int(1+nL*time_unitary/timeStep))*timeStep

    if init_eigenstate=='up':
        rho0 = rho_up
    elif init_eigenstate=='down':
        rho0 = rho_down
    else:
        rho0 = rho_left

    # results=[]
    # for repetition in range(n_rep):
    #     rho_vec = threeLvls_DissDyn(rho0,time_unitary,timeStep,nL,prob_abs,ν,δp_factor,δm_factor,ϕ,beta,digitsPrecision=digitsPrecision)
    #     results.append(rho_vec)
    # results = np.array(results)
    results=0
    for repetition in range(n_rep):
        rho_vec = threeLvls_DissDyn(rho0,time_unitary,timeStep,nL,prob_abs,ν,δp_factor,δm_factor,ϕ,beta,digitsPrecision=digitsPrecision)
        results += rho_vec
    timeVector = timeVector*1e-3 # to have the time in μs

    rho_mean_Z = results/n_rep
    if not(change_basis):
        return timeVector,rho_mean_Z
    else: #change_basis=True
        rho_mean_H = np.zeros(rho_mean_Z.shape,dtype=complex)
        for i in range(len(timeVector)):
            rho_new = rho_mean_Z[i]
            rho_mean_H[i] = np.round(np.matmul(p_mat.conjugate().transpose(),np.matmul(rho_new,p_mat)),digitsPrecision)
        return timeVector,rho_mean_H
