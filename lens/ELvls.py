
import numpy as np
from qutip import *
# I will use the solver from http://qutip.org/docs/3.1.0/guide/dynamics/dynamics-master.html
# (see the section **The Lindblad Master equation**)

#
### Definition of constants:
#
# Quantities for the Hamiltonian (all in MHz)
γe = 2.802495164;
γn = -3.08e-4;
Δg = 2.87036e3;
Pg = -4.945; #-4.935 ? EXP.
Ag = -2.162;
Bg = -2.62;
Δe = 1420;
Pe = -4.945;
Ae = -40;
Be = -21.25;
#
# Definition of 3-lvl 'pauli' matrices
x1=(1/np.sqrt(2))*np.array([[0,1,0],[1,0,1],[0,1,0]])
y1=(1j/np.sqrt(2))*np.array([[0,-1,0],[1,0,-1],[0,1,0]])
z1=np.array([[1,0,0],[0,0,0],[0,0,-1]])
#
# And some other useful matrices
Id3 = np.identity(3)
em = np.array([[0, 0], [0, 1]]);
ep = np.array([[1, 0], [0, 0]]);
#
# Dimensions of the Hilbert space
Nn = 3;
Ng = 3*Nn;
n = Nn + 2*Ng;


### Definition of useful functions:
#
# Hamiltonians of the Ground and Excited states (separately)
def Hg(B, θ):
    return Δg*np.kron(z1*z1, Id3) + γe*B*np.kron(np.cos(θ)*z1 + np.sin(θ)*x1, Id3) + γn*B*np.kron(Id3,np.cos(θ)*z1 + np.sin(θ)*x1) + Pg*np.kron(Id3,z1*z1) + Ag*np.kron(z1,z1) + Bg*(np.kron(x1,x1) + np.kron(y1,y1))
def He(B, θ):
    return Δe*np.kron(z1*z1, Id3) + γe*B*np.kron(np.cos(θ)*z1 + np.sin(θ)*x1, Id3) + γn*B*np.kron(Id3,np.cos(θ)*z1 + np.sin(θ)*x1) + Pe*np.kron(Id3,z1*z1) + Ae*np.kron(z1,z1) + Be*(np.kron(x1,x1) + np.kron(y1,y1))
# Total Hamiltonian
# Ordine elementi di matrice + 1, 0 , -1
def Htot(B, θ):
    tM = np.zeros([n,n],dtype=complex)
    tM[:2*Ng,:2*Ng] = np.kron(ep, Hg(B, θ)) + np.kron(em, He(B, θ))
    return Qobj(tM)
#
# Function to connect states (decay rates)
# Notation: Ljk(finale, iniziale, dim)
def Ljk(j, k, n):
    tM = np.zeros([n,n])
    tM[j,k]=1
    return tM
#
# Collection of Lindblad terms
# This will be the 'c_ops' argument in the QuTiP function 'mesolve'
def ldb(W,drates,Nn=Nn,Ng=Ng,n=n):
    Γ,Γ1m,Γ0m,Γm0,Γm1,theta = drates
    # list of decays:
    ll = list(Qobj(np.sqrt(Γ*np.cos(theta)**2)*Ljk(j,j+Ng,n)) for j in range(Ng))
    ll.extend( list(Qobj(np.sqrt(Γ*1.0*np.sin(theta)**2)*Ljk(j + Nn, j + Ng, n)) for j in range(Nn)) )
    ll.extend( list(Qobj(np.sqrt(Γ*0.5*np.sin(theta)**2)*Ljk(j + 2*Nn, j + Ng + Nn, n)) for j in range(Nn)) )
    ll.extend( list(Qobj(np.sqrt(Γ*0.5*np.sin(theta)**2)*Ljk(j, j + Ng + Nn, n)) for j in range(Nn)) )
    ll.extend( list(Qobj(np.sqrt(Γ*1.0*np.sin(theta)**2)*Ljk(j + Nn, j + Ng + 2*Nn, n)) for j in range(Nn)) )
    # 2 out of 4 have a 0.5 because there are 2 possible forbidden routes to decay from the ms=0 excited state
    ll.extend( list(Qobj(np.sqrt(W*Γ)*Ljk(j + Ng, j, n)) for j in range(Ng)) )
    ll.extend( list(Qobj(np.sqrt(W*Γ*0.5*np.sin(theta)**2)*Ljk(j + Ng + Nn, j, n)) for j in range(Nn)) )
    ll.extend( list(Qobj(np.sqrt(W*Γ*1.0*np.sin(theta)**2)*Ljk(j + Ng + 2*Nn, j + Nn, n)) for j in range(Nn)) )
    ll.extend( list(Qobj(np.sqrt(W*Γ*1.0*np.sin(theta)**2)*Ljk(j + Ng, j + Nn, n)) for j in range(Nn)) )
    ll.extend( list(Qobj(np.sqrt(W*Γ*0.5*np.sin(theta)**2)*Ljk(j + Ng + Nn, j + 2*Nn, n)) for j in range(Nn)) )
    # The 0.5 is because there are two possible forbidden excitation routes from the ms=0 ground state
    ll.extend( list(Qobj(np.sqrt(Γ1m)*Ljk(2*Ng + j, j + Ng, n)) for j in range(Nn)) )
    ll.extend( list(Qobj(np.sqrt(Γ0m)*Ljk(2*Ng + j, j + Ng + Nn, n)) for j in range(Nn)) )
    ll.extend( list(Qobj(np.sqrt(Γ1m)*Ljk(2*Ng + j, j + 2*Ng - Nn, n)) for j in range(Nn)) )
    ll.extend( list(Qobj(np.sqrt(Γm1)*Ljk(j, j + 2*Ng, n)) for j in range(Nn)) )
    ll.extend( list(Qobj(np.sqrt(Γm0)*Ljk(j + Nn, j + 2*Ng, n)) for j in range(Nn)) )
    ll.extend( list(Qobj(np.sqrt(Γm1)*Ljk(j + 2*Nn, j + 2*Ng, n)) for j in range(Nn)) )
    return ll
#
# Matrix M with M[n,n] = 1 and all other entries equal to zero
# Used for the definition of the observables 'eops'
def st(nn,n=n):
    tM = np.zeros([n,n])
    tM[nn,nn] = 1
    return tM


### Decay rates (all in MHz):
#
## Parameters from Robledo.1 -- Robledo L., et al, New J. Phys. 13 025013 (2011)
ΓRob = 63.48
Γ1mRob = 79.91
Γ0mRob = 11.76
Γm0Rob = 3.25
Γm1Rob = 2.37
drRob = (ΓRob,Γ1mRob,Γ0mRob,Γm0Rob,Γm1Rob,0)

## Parameters from Robledo.2 -- Robledo L., et al, New J. Phys. 13 025013 (2011)
ΓRob2 = 65.08
Γ1mRob2 = 79.83
Γ0mRob2 = 10.56
Γm0Rob2 = 3.0
Γm1Rob2 = 2.61
drRob2 = (ΓRob2,Γ1mRob2,Γ0mRob2,Γm0Rob2,Γm1Rob2,0)

## Parameters from Tetienne -- Tetienne J.-P., et al, New J. Phys. 14 103033 (2012)
ΓTet = 67.7
Γ1mTet = 6.4
Γ0mTet = 50.7
Γm0Tet = 0.7
Γm1Tet = 0.6
drTet = (ΓTet,Γ1mTet,Γ0mTet,Γm0Tet,Γm1Tet,0)

## Parameters from Manson -- Manson N. B., et al, Phys. Rev. B 74 104303 (2006)
ΓMan = 77;
Γ1mMan = 0.35*77; #27;
Γ0mMan = 0.0;
Γm0Man = 1/0.3; #3.3;
Γm1Man = 0.0;
drMan = (ΓMan,Γ1mMan,Γ0mMan,Γm0Man,Γm1Man,0)

## Phenomenological
ΓPhe = 77;
Γ1mPhe = 77;
Γ0mPhe = 0.0;
Γm0Phe = 33.33;
Γm1Phe = 0.0;
drPhe = (ΓPhe,Γ1mPhe,Γ0mPhe,Γm0Phe,Γm1Phe,0)

## Parameters from Wolters -- Janik Wolters, et al, Phys. Rev. A 88 020101(R) (2013)
ΓWol = 77;
Γ1mWol = 62; #27;
Γ0mWol = 3;
Γm0Wol = 4.55; #3.3;
Γm1Wol = 0.0;
thWol = 12.4*np.pi/180;
drWol = (ΓWol,Γ1mWol,Γ0mWol,Γm0Wol,Γm1Wol,thWol)

## Parameters from our experiment (only for the optical model) -- see LabLog  2019_February_11
ΓDqt = 77;
Γ1mDqt = 60.4; #62;
Γ0mDqt = 9.39; #9.4;
Γm0Dqt = 9.6; #9.6;
Γm1Dqt = 0.0;
thDqt = 0.193; #~11° #0.183; #~10.5°
drDqt = (ΓDqt,Γ1mDqt,Γ0mDqt,Γm0Dqt,Γm1Dqt,thDqt)

### Parameters for the simulation:
#
# This is the 'e_ops' argument in the QuTiP function 'mesolve'.
# It can be a list of operators for which to calculate the expectation value.
# Here we calculate each of the elements of the diagonal (all the populations separately).
#
# This is for calculating all rho
def eopsRho(n=n):
    return [Qobj(np.reshape(np.eye(1,n*n,i),(n,n))) for i in range(n*n)]
#
# This is for calculating all the populations
def eopsPopAll(n=n):
    return [Qobj(st(i,n=n)) for i in range(n)]
#
# This is for calculating the populations of the excited states
def eopsPopExc(Ng=Ng,n=n):
    return [Qobj(st(i,n=n)) for i in (np.arange(Ng)+Ng)]
#
# This is for calculating the populations of the ms=0 ground state (three hyperfine states)
def eopsPopGms0(Nn=Nn,n=n):
    return [Qobj(st(i,n=n)) for i in (np.arange(Nn)+Nn)]


### Function to run the simulation
#
def eLvls(bAmplitude,bAngle,psi0,times,er,gRates,obs):
    '''
    :param bAmplitude: magnetic field strength (float).
    :param bAngle:  magnetic field angle (float).
    :param psi0:    initial state (rho0 -> (nxn) array; psi0 (nx1) array).
    :param times:   time vector (one dimension np.array).
    :param er:  excitation rate, in terms of the radiative decay rate: er x Γ (float).
    :param gRates:  decay rates in order: list(Γ,Γ1m,Γ0m,Γm0,Γm1);
                    or name of preloaded models: 'Wol', 'Rob', 'Rob2', 'Tet', 'Man' or 'Phe'.
    :param obs:    list of operators to obtain the expected value (list of (nxn) arrays);
                    or name of preloaded operators: 'rho' or 'popul' or 'populE' or 'populGms0'.
    :returns: list of expected values, each element is a np.array with the solution for each different time.
    '''
    # Selecting the decay rates model. If not 'Rob', 'Man' or 'Phe', it should be a list (Γ,Γ1m,Γ0m,Γm0,Γm1)
    if gRates == 'Rob':
        gRates = drRob
    elif gRates == 'Rob2':
        gRates = drRob2
    elif gRates == 'Tet':
            gRates = drTet
    elif gRates == 'Man':
        gRates = drMan
    elif gRates == 'Phe':
        gRates = drPhe
    elif gRates == 'Wol':
        gRates = drWol
    #
    # Selecting the list of operators to calculate the expectation values
    if obs == 'rho':
        eops = eopsRho()
    elif obs == 'popul':
        eops = eopsPopAll()
    elif obs == 'populE':
        eops = eopsPopExc()
    elif obs == 'populGms0':
        eops = eopsPopGms0()
    else:
        eops = [Qobj(oper) for oper in obs]
    #
    return mesolve(Htot(bAmplitude,bAngle), Qobj(psi0) , times, ldb(er,gRates), eops)



###
###
###
### Reduced system (no hyperfine)
###
Nn_R = 1;
Ng_R = 3*Nn_R;
n_R = Nn_R + 2*Ng_R;
### Definition of useful functions:
#
# Hamiltonians of the Ground and Excited states (separately)
def Hg_R(B, θ):
    return Δg*(z1*z1) + γe*B*(np.cos(θ)*z1 + np.sin(θ)*x1)
def He_R(B, θ):
    return Δe*(z1*z1) + γe*B*(np.cos(θ)*z1 + np.sin(θ)*x1)
# Total Hamiltonian
# Ordine elementi di matrice + 1, 0 , -1
def Htot_R(B, θ):
    tM = np.zeros([n_R,n_R],dtype=complex)
    tM[:2*Ng_R,:2*Ng_R] = np.kron(ep, Hg_R(B, θ)) + np.kron(em, He_R(B, θ))
    return Qobj(tM)
#
### Function to run the simulation
#
def eLvls_R(bAmplitude,bAngle,psi0,times,er,gRates,obs):
    '''
    :param bAmplitude: magnetic field strength (float).
    :param bAngle:  magnetic field angle (float).
    :param psi0:    initial state (rho0 -> (nxn) array; psi0 (nx1) array).
    :param times:   time vector (one dimension np.array).
    :param er:  excitation rate, in terms of the radiative decay rate: er x Γ (float).
    :param gRates:  decay rates in order: list(Γ,Γ1m,Γ0m,Γm0,Γm1);
                    or name of preloaded models: 'Wol', 'Rob', 'Rob2', 'Tet', 'Man' or 'Phe'.
    :param obs:    list of operators to obtain the expected value (list of (nxn) arrays);
                    or name of preloaded operators: 'rho' or 'popul' or 'populE' or 'populGms0'.
    :returns: list of expected values, each element is a np.array with the solution for each different time.
    '''
    # Selecting the decay rates model. If not 'Rob', 'Man' or 'Phe', it should be a list (Γ,Γ1m,Γ0m,Γm0,Γm1)
    if gRates == 'Rob':
        gRates = drRob
    elif gRates == 'Rob2':
        gRates = drRob2
    elif gRates == 'Tet':
        gRates = drTet
    elif gRates == 'Man':
        gRates = drMan
    elif gRates == 'Phe':
        gRates = drPhe
    elif gRates == 'Wol':
        gRates = drWol
    #
    # Selecting the list of operators to calculate the expectation values
    if obs == 'rho':
        eops = eopsRho(n_R)
    elif obs == 'popul':
        eops = eopsPopAll(n_R)
    elif obs == 'populE':
        eops = eopsPopExc(Ng_R,n_R)
    elif obs == 'populGms0':
        eops = eopsPopGms0(Nn_R,n_R)
    else:
        eops = [Qobj(oper) for oper in obs]
    #
    return mesolve(Htot_R(bAmplitude,bAngle), Qobj(psi0) , times, ldb(er,gRates,Nn_R,Ng_R,n_R), eops)
#
### Including interaction with MW:
#
# Definition of 2-lvl 'pauli' matrices for ms = 0 and ms = +1
x1d2=(1/2)*np.array([[0,1,0],[1,0,0],[0,0,0]])
y1d2=(1j/2)*np.array([[0,-1,0],[1,0,0],[0,0,0]])
z1d2=(1/2)*np.array([[1,0,0],[0,-1,0],[0,0,0]])
# Total Hamiltonian with MW interaction
def Htot_MW_R(Ωx, Ωy, δ): #B, θ,
    tM = np.zeros([n_R,n_R],dtype=complex)
    Hmw = Ωx*x1d2 + Ωy*y1d2 + δ*z1d2
    #tM[:2*Ng_R,:2*Ng_R] = np.kron(ep, Hg_R(B, θ) + Hmw) + np.kron(em, He_R(B, θ))
    tM[:2*Ng_R,:2*Ng_R] = np.kron(ep, Hmw)
    return Qobj(tM)
#
### Function to run the simulation
#
def eLvls_MW_R(psi0,times,er,gRates,obs,Ωx, Ωy, δ): #bAmplitude,bAngle,
    '''
    :param psi0:    initial state (rho0 -> (nxn) array; psi0 (nx1) array).
    :param times:   time vector (one dimension np.array).
    :param er:      excitation rate, in terms of the radiative decay rate: er x Γ (float).
    :param gRates:  decay rates in order: list(Γ,Γ1m,Γ0m,Γm0,Γm1);
                    or name of preloaded models: 'Wol', 'Rob', 'Rob2', 'Tet', 'Man' or 'Phe'.
    :param obs:    list of operators to obtain the expected value (list of (nxn) arrays);
                    or name of preloaded operators: 'rho' or 'popul' or 'populE' or 'populGms0'.
    :param Ωx:      Rabi frequency (on resonance), X component.
    :param Ωy:      Rabi frequency (on resonance), Y component.
    :param δ:       Detuning of the MW.
    :returns: list of expected values, each element is a np.array with the solution for each different time.
    '''
    # Selecting the decay rates model. If not 'Rob', 'Man' or 'Phe', it should be a list (Γ,Γ1m,Γ0m,Γm0,Γm1)
    if gRates == 'Rob':
        gRates = drRob
    elif gRates == 'Rob2':
        gRates = drRob2
    elif gRates == 'Tet':
        gRates = drTet
    elif gRates == 'Man':
        gRates = drMan
    elif gRates == 'Phe':
        gRates = drPhe
    elif gRates == 'Wol':
        gRates = drWol
    #
    # Selecting the list of operators to calculate the expectation values
    if obs == 'rho':
        eops = eopsRho(n_R)
    elif obs == 'popul':
        eops = eopsPopAll(n_R)
    elif obs == 'populE':
        eops = eopsPopExc(Ng_R,n_R)
    elif obs == 'populGms0':
        eops = eopsPopGms0(Nn_R,n_R)
    else:
        eops = [Qobj(oper) for oper in obs]
    #
    return mesolve(Htot_MW_R(Ωx, Ωy, δ), Qobj(psi0) , times, ldb(er,gRates,Nn_R,Ng_R,n_R), eops)

# Total Hamiltonian with MW interaction
def Htot_MW_R_opt(Ωx, Ωy, δ, oΩ,theta): #B, θ,
    # The MW coherent coupling part:
    tM = np.zeros([n_R,n_R],dtype=complex)
    Hmw = Ωx*x1d2 + Ωy*y1d2 + δ*z1d2
    tM[:2*Ng_R,:2*Ng_R] = np.kron(ep, Hmw)
    # The optical transition part:
    opt = sum( list((np.cos(theta)**2)*Ljk(j + Ng_R, j, n_R) for j in range(Ng_R)) )
    forb1 = sum( list((0.5*np.sin(theta)**2)*Ljk(j + Nn_R, j + Ng_R, n_R) for j in range(Nn_R)) )
    forb2 = sum( list((1.0*np.sin(theta)**2)*Ljk(j + 2*Nn_R, j + Ng_R + Nn_R, n_R) for j in range(Nn_R)) )
    forb3 = sum( list((1.0*np.sin(theta)**2)*Ljk(j, j + Ng_R + Nn_R, n_R) for j in range(Nn_R)) )
    forb4 = sum( list((0.5*np.sin(theta)**2)*Ljk(j + Nn_R, j + Ng_R + 2*Nn_R, n_R) for j in range(Nn_R)) )
    # The 0.5 is because there are two possible forbidden excitation routes from the ms=0 ground state
    return Qobj(tM) + oΩ*Qobj((opt + opt.transpose()) + ((forb1+forb2+forb3+forb4)+(forb1+forb2+forb3+forb4).transpose()))
#
### Function to run the simulation
#
def eLvls_MW_R_opt(psi0,times, optΩ,gRates,obs,Ωx, Ωy, δ): #bAmplitude,bAngle,
    '''
    :param psi0:    initial state (rho0 -> (nxn) array; psi0 (nx1) array).
    :param times:   time vector (one dimension np.array).
    :param optΩ:    excitation rate, in terms of the radiative decay rate: optΩ x Γ (float).
    :param gRates:  decay rates in order: list(Γ,Γ1m,Γ0m,Γm0,Γm1);
                    or name of preloaded models: 'Wol', 'Rob', 'Rob2', 'Tet', 'Man' or 'Phe'.
    :param obs:    list of operators to obtain the expected value (list of (nxn) arrays);
                    or name of preloaded operators: 'rho' or 'popul' or 'populE' or 'populGms0'.
    :param Ωx:      Rabi frequency (on resonance), X component.
    :param Ωy:      Rabi frequency (on resonance), Y component.
    :param δ:       Detuning of the MW.

    :returns: list of expected values, each element is a np.array with the solution for each different time.
    '''
    # Selecting the decay rates model. If not 'Rob', 'Man' or 'Phe', it should be a list (Γ,Γ1m,Γ0m,Γm0,Γm1)
    if gRates == 'Rob':
        gRates = drRob
    elif gRates == 'Rob2':
        gRates = drRob2
    elif gRates == 'Tet':
        gRates = drTet
    elif gRates == 'Man':
        gRates = drMan
    elif gRates == 'Phe':
        gRates = drPhe
    elif gRates == 'Wol':
        gRates = drWol
    elif gRates == 'Dqt':
        gRates = drDqt
    #
    # Selecting the list of operators to calculate the expectation values
    if obs == 'rho':
        eops = eopsRho(n_R)
    elif obs == 'popul':
        eops = eopsPopAll(n_R)
    elif obs == 'populE':
        eops = eopsPopExc(Ng_R,n_R)
    elif obs == 'populGms0':
        eops = eopsPopGms0(Nn_R,n_R)
    else:
        eops = [Qobj(oper) for oper in obs]
    #
    return mesolve(Htot_MW_R_opt(Ωx,Ωy,δ,gRates[0]*optΩ,gRates[5]), Qobj(psi0) , times, ldb(0,gRates,Nn_R,Ng_R,n_R), eops)
