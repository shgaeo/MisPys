import numpy as np

#Definition of Pauli matrices:
id2=np.array([[1,0],[0,1]])
x=np.array([[0,1],[1,0]])
y=np.array([[0,-1j],[1j,0]])
z=np.array([[1,0],[0,-1]])

#fast checks on Pauli matrices
involutory = np.array_equal(np.matmul(z,z),id2)&np.array_equal(np.matmul(x,x),id2)&np.array_equal(np.matmul(y,y),id2)
determinant = (np.linalg.det(z)==-1) & (np.linalg.det(x)==-1) & (np.linalg.det(y)==-1)
trace = (np.trace(z)==0)&(np.trace(x)==0)&(np.trace(y)==0)
if not( involutory&determinant&trace ):
    print('Error: Something is wrong with the definition of the Pauli matrices')

def cpmg(t1,nN,rm=1,mM=1):
    seq=np.ones(nN+1)*2*t1
    seq[0]=t1
    seq[-1]=t1
    return seq

def nested(tT,nN,rm,mM):
    #  Define the spacing between pi-pulses
    distPi = np.ones([mM,nN])
    for j in range(1,nN+1):
        for h in range(1,mM+1):
            rh = (2*h-mM-1)/(2*mM);
            distPi[h-1,j-1] = ((2*j-1)/2+rm*rh)/nN
    distPi=distPi.flatten('F') # to turn it into a vector of nxm
    seq=np.ones(nN*mM+1)
    seq[:-1]=distPi
    seq[-1]=1
    seq[1:]=seq[1:]-seq[:-1]
    return tT*seq

def nested2(tT,nN,rm,mM): #same as nested but with one pi pulse between nests
    #  Define the spacing between pi-pulses
    distPi = np.ones([mM+1,nN])
    for j in range(1,nN+1):
        for h in range(1,mM+1):
            rh = (2*h-mM-1)/(2*mM);
            distPi[h-1,j-1] = ((2*j-1)/2+rm*rh)/nN
        if j!=nN:
            distPi[mM,j-1]=j/nN
    distPi=distPi.flatten('F') # to turn it into a vector of nxm
    distPi=distPi[:-1]
    seq=np.ones(nN*mM+1+nN-1)
    seq[:-1]=distPi
    seq[-1]=1
    seq[1:]=seq[1:]-seq[:-1]
    return tT*seq

def uhrig(tT,nN,rm=1,mM=1):
    seq=np.ones(nN+1)
    for jj in range(1,nN+2):
        seq[jj-1] = np.sin(np.pi*jj/(2*nN+2))**2 - np.sin(np.pi*(jj-1)/(2*nN+2))**2
    return tT*seq

def nestedUhrig(tT,nN,rm=1,mM=1):
    #  Define the spacing between pi-pulses
    distPi = np.ones([mM,nN])
    for j in range(1,nN+1):
        if mM==1:
            distPi[0,j-1] = (np.sin(np.pi*j/(2*nN+2)))**2
        else:
            for h in range(1,mM+1):
                rh= (np.sin(np.pi/(2*nN+2)))**2*(2*h-mM-1)/(mM-1)
                distPi[h-1,j-1] = (np.sin(np.pi*j/(2*nN+2)))**2 + rm*rh
    distPi=distPi.flatten('F') # to turn it into a vector of nxm
    seq=np.ones(nN*mM+1)
    seq[:-1]=distPi
    seq[-1]=1
    seq[1:]=seq[1:]-seq[:-1]
    return tT*seq


def mMtv(tVec,nN,pro0,pro1,rm=1,mM=1):
    if len(tVec)-1!=nN*mM:
        if len(tVec)-1!=(nN*mM+nN-1):
            print('Error: len(tVec) must be equal to nN*mM+1 (CPMG,Uhrig,Nested) or nN*mM+1+nN-1 (Nested2)')
            return -1
        #print('Error: len(tVec) must be equal to nN*mM+1')
        #return -1
    u0temp=id2
    u1temp=id2
    for i in range(len(tVec)): #range(nN*mM+1):
        if i%2==0:
            u0temp = np.matmul( pro0(tVec[i]) , u0temp )
            u1temp = np.matmul( pro1(tVec[i]) , u1temp )
        else:
            u0temp = np.matmul( pro1(tVec[i]) , u0temp )
            u1temp = np.matmul( pro0(tVec[i]) , u1temp )
    resp=np.trace( np.matmul( u0temp , (u1temp.T).conjugate() ) )
    if resp.imag!=0:
        print('Warning: M has imaginary part')
    return (1/2)*resp.real

def mM(tt,nN,A,B,wL,funcTVec,rm=1,mM=1):
    wT = np.sqrt((A + wL)**2 + B**2)
    mx = B/wT
    mz = (A + wL)/wT

    def pro0(t):
        return id2*np.cos(wL*t/2) - 1j*z*np.sin(wL*t/2)
    def pro1(t):
        return id2*np.cos(wT*t/2) - 1j*(mx*x + mz*z)*np.sin(wT*t/2)

    res=np.zeros(len(tt))
    for i in range(len(tt)):
        tVec = funcTVec(tt[i],nN,rm,mM)
        res[i]= mMtv(tVec,nN,pro0,pro1,rm,mM)
    return res
