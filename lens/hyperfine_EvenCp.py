# From PhysRevLett.109.137602(2012)

import numpy as np


def wtil(B, A,wL):
    return np.sqrt((A + wL)**2 + B**2)
def mx(B, A,wL):
    return B/wtil(B,A,wL)
def mz(B, A,wL):
    return (A + wL)/wtil(B,A,wL)

def alpha(t,B, A,wL):
    #if round(mx(Btemp, A=A1,wL=ωL)**2 + mz(Btemp)**2,12)!=1:
    if round(mx(B,A,wL)**2 + mz(B,A,wL)**2,12)!=1:
        print('ERROR in \alpha: mx²+mz²!=1')
        return -1
    return wtil(B,A,wL)*t
def beta(t, wL):
    return wL*t

def phi(t,B, A,wL):
    return np.pi-np.arccos(np.cos(alpha(t,B,A,wL))*np.cos(beta(t,wL)) - mz(B,A,wL)*np.sin(alpha(t,B,A,wL))*np.sin(beta(t,wL)))
def phi_original(t,B, A,wL):
    return np.arccos(np.cos(alpha(t,B,A,wL))*np.cos(beta(t,wL)) - mz(B,A,wL)*np.sin(alpha(t,B,A,wL))*np.sin(beta(t,wL)))
def n0n1(t,B, A,wL):
    return 1 - mx(B,A,wL)**2*( (1 - np.cos(alpha(t,B,A,wL)))*(1 - np.cos(beta(t,wL))) )/( 1 - np.cos(phi(t,B, A,wL)))
def n0n1_original(t,B, A,wL):
    return 1 - mx(B,A,wL)**2*( (1 - np.cos(alpha(t,B,A,wL)))*(1 - np.cos(beta(t,wL))) )/( 1 + np.cos(phi(t,B, A,wL)))


def M(t,B, A,wL,nNn):
    return 1 - (1 - n0n1(t,B,A,wL))*np.sin(nNn*phi(t,B,A,wL)/2)**2

#def P(t,B, A=A1,wL=ωL,nNn=nN):
#    return (M(t,B,A,wL,nNn) + 1)/2

#probability including the T2L decay (NOTE: in terms of N not in terms of time)
def Pn(nt2l,t,B, A,wL,nNn):
    return (M(t,B,A,wL,nNn)*np.exp(-nNn/nt2l) + 1)/2


def tk(k,A,wL):
    return np.pi*(2*k - 1)/(2*wL + A)
