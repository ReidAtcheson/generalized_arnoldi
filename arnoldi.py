import numpy as np
import scipy.linalg as la


def arnoldi(A,v,k):
    norm=np.linalg.norm
    dot=np.dot
    eta=1.0/np.sqrt(2.0)

    m=len(v)
    V=np.zeros((m,k+1))
    H=np.zeros((k+1,k))
    #V[:,0]=v/norm(v)
    V[:,0]=v/norm(v)
    for j in range(0,k):
        w=A(V[:,j])
        h=V[:,0:j+1].T @ w
        f=w-V[:,0:j+1] @ h
        s = V[:,0:j+1].T @ f
        f = f - V[:,0:j+1] @ s
        h = h + s
        beta=norm(f)
        #H[j+1,j]=beta
        H[0:j+1,j]=h
        H[j+1,j]=beta
        #V[:,j+1]=f/beta
        V[:,j+1]=f.flatten()/beta
    return V,H

