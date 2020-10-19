# In this script there is a function to calculate the Hamiltonian of the ground state
# of an NV center, including the Nitrogen hyperfine coupling


import numpy as np
from scipy.linalg import eigvals
#
from lens.analysisFunctions import fit_func

γe = 2.802495164;
γn = -3.08*1e-4;
Ag = -2.162;
Bg = -2.62;
Dg = 2.87048*1e3; #2.86928*1e3; #2.87*1e3;
#DDg = 0.36;
Qg = -4.945;
De = 1420;
Pe = -4.945;
Ae = -40;
Be = -58;

# Definition of 3x3 Pauli matrices and the identity
x1 = np.array([[0,1,0],[1,0,1],[0,1,0]])/np.sqrt(2)
y1 = np.array([[0,-1,0],[1,0,-1],[0,1,0]])*1j/np.sqrt(2)
z1 = np.array([[1,0,0],[0,0,0],[0,0,-1]])
id1 = np.array([[1,0,0],[0,1,0],[0,0,1]])

#np.allclose(np.dot(x1,np.dot(x1,x1)),x1),np.allclose(np.dot(y1,np.dot(y1,y1)),y1),np.allclose(np.dot(z1,np.dot(z1,z1)),z1)

# Function to calculate the Hamiltonian in terms of the angle and the field strength
def H(θ,B):
    he = Dg*np.kron(np.dot(z1,z1), id1) + γe*B*np.kron(np.cos(θ*np.pi/180)*z1 + np.sin(θ*np.pi/180)*x1, id1)

    #hi = (Ag/2)*np.kron(z1, id1) + np.sqrt(2)*Ag*np.kron(z1, z1) + Bg*(np.kron(x1, x1) + np.kron(y1, y1))
    hi = Ag*np.kron(z1, z1) + Bg*(np.kron(x1, x1) + np.kron(y1, y1))

    hn = Qg*np.kron(id1, np.dot(z1,z1)) + γn*B*np.kron(id1,np.cos(θ*np.pi/180)*z1 + np.sin(θ*np.pi/180)*x1)
    return he + hi + hn

# Function to calculate the electronic spin transition frequencies
def transitionsFreq(θ,B):
    eVals = np.sort(eigvals(H(θ,B)).real)
    e0p1,e0m1,e00 = eVals[:3]     # | 0,+1> , | 0,-1> , | 0,0>
    em1m1,em1p1,em10 = eVals[3:6] # |-1,-1> , |-1,+1> , |-1,0>
    ep1p1,ep1m1,ep10 = eVals[6:9] # |+1,+1> , |+1,-1> , |+1,0>
    return em1m1-e0m1,em10-e00,em1p1-e0p1,  ep1p1-e0p1,ep10-e00,ep1m1-e0m1

# Fitting the parameters: angle and field strength
def func2fit(x,theta,B):
    tf = np.array(transitionsFreq(theta,B))
    return tf[x]
def fromRamseyFitField(freqsExper,freqsExperE,p0):
    para,perr,r2 = fit_func(func2fit,np.arange(6,dtype=int),freqsExper,p0,yderr=freqsExperE)
    return para,perr,r2
