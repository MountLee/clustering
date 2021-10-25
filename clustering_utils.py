import numpy as np
import matplotlib.pyplot as plt

def visual_result(x, idx):
    center = np.unique(idx)
    k = len(center)
    for i in range(k):
        ii = np.where(idx == center[i])[0]
        col = np.random.rand(3)
        plt.plot(x[ii,0],x[ii,1],'o',color = col)

def visual_result_repre(x, idx):
    center = np.unique(idx)
    k = len(center)
    for i in range(k):
        ii = np.where(idx == center[i])[0]
        col = np.random.rand(3)
        for j in ii:
            plt.plot([x[j,0],x[center[i],0]],
                     [x[j,1],x[center[i],1]],'-o',color = col)

def ap_cluster(S, p, symmetric = True, nonoise = False,
    maxits = 1000, convits = 100, lam = 0.7, details = False, plt = False):
    """
    affinity propagation clustering

    S (N-by-N array): similarity matrix of N points
    p (N-by-. array): p[i] is preference that point i be chosen as a center
    

    ###### TO DO:
    implement
        details
        plt
        sparse version
    """
    N, _ = S.shape
    realmin, realmax = -1E300, 1E300

    for i in range(N):
        S[i,i] = p[i]

    dS = p.copy()
    A, R = np.zeros((N,N)), np.zeros((N,N))
    t = 1

    dn = 0
    i = 0
    e = np.zeros((N, convits))

    if symmetric:
        ST = S
    else:
        ST = S.T

    while not dn:
        i += 1

        # compute responsibilities
        A, R = A.T, R.T
        for ii in range(N):
            old = R[:,ii].copy()
            AS = A[:,ii] + ST[:,ii]
            Y, I = AS.max(), AS.argmax()
            AS[I] = -np.inf
            Y2, I2 = AS.max(), AS.argmax()
            R[:,ii] = ST[:,ii] - Y
            R[I,ii] = ST[I,ii] - Y2
            R[:,ii] = (1 - lam) * R[:,ii] + lam * old
            R[R[:,ii] > realmax, ii] = realmax
        A, R = A.T, R.T
        
        # compute availabilities
        for jj in range(N):
            old = A[:,jj].copy()
            Rp = np.maximum(R[:,jj],0)
            Rp[jj] = R[jj,jj]
            A[:,jj] = sum(Rp) - Rp
            dA = A[jj,jj]
            A[:,jj] = np.minimum(A[:,jj], 0)
            A[jj,jj] = dA
            A[:,jj] = (1 - lam) * A[:,jj] + lam * old

        # check for convergence
        E = ((np.diag(A).copy() + np.diag(R).copy()) > 0) * 1
        e[:, (i - 1) % convits] = E
        K = sum(E)
        if i >= convits or i >= maxits:
            se = np.sum(e,axis = 1)
            unconverged = (sum((se == convits) + (se == 0)) != N)
            if (not unconverged and K > 0) or i == maxits:
                dn = 1

    # identify centers and clusters
    I = np.where((np.diag(A) + np.diag(R)) > 0)[0]
    K = len(I)
    print(K)
    if K > 0:
        c = np.argmax(S[:,I], axis = 1)
        tmp = S[np.arange(N),c]
        c[I] = np.arange(K)
        for k in range(K):
            ii = np.where(c == k)[0]
            sub = S[:,ii][ii,:]
            j = np.argmax(sub, axis = 0)
            y = sub[j,np.arange(len(ii))]
            I[k] = ii[j[0]]
        notI = np.setdiff1d(np.arange(N), I)
        c = np.argmax(S[:,I], axis = 1)
        tmp = S[np.arange(N),c]
        c[I] = np.arange(K)
        tmpidx = I[c]
        tmpdpsim = sum(S[notI,tmpidx[notI]])
        tmpexpref = sum(dS[I])
        tmpnetsim = tmpdpsim + tmpexpref
    else:
        tmpnetsim = np.nan
        tmpexpref = np.nan
        tmpdpsim = np.nan
        tmpidx = np.array([np.nan for i in range(N)])
    #if details:
    netsim = tmpnetsim
    dpsim = tmpdpsim
    expref = tmpexpref
    idx = tmpidx
    return idx, netsim, dpsim, expref, A, R