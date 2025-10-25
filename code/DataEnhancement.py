import numpy as np



def SVD(Y):
    #SVD
    U, S, V = np.linalg.svd(Y)
    S=np.sqrt(S)
    r = 30
    Wt = np.zeros([r,r])
    for i in range(0,r):
        Wt[i][i]=S[i]
    U= U[:, 0:r]
    V= V[0:r,:]
    A = np.dot(U,Wt)
    B1 = np.dot(Wt,V)
    B=np.transpose(B1)
    Y = A @ B.T

    return Y


