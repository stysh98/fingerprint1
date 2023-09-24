import numpy as np
import scipy.optimize

def mean_and_covarianceMatrix(D):
    N = D.shape[1]
    av = D.mean(1)
    mu = av.reshape((av.size, 1))
    DC = D - mu
    C = np.dot(DC, DC.T)/N
    return mu, C

def logpdf_GAU_ND(X,mu,C) :
    res = -0.5*X.shape[0]*np.log(2*np.pi)
    res += -0.5*np.linalg.slogdet(C)[1]
    res += -0.5*((X-mu)*np.dot(np.linalg.inv(C), (X-mu))).sum(0) 
    return res

def logpdf_GMM(X, gmm):
    SJ = np.zeros((len(gmm),X.shape[1]))
    
    for g, (w, mu, C) in enumerate(gmm):
        SJ[g,:] = logpdf_GAU_ND(X, mu, C) + np.log(w)

    SM = scipy.special.logsumexp(SJ, axis=0)
    
    return SJ, SM 

def GMM_EM(X, gmm):
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]
    
    psi = 0.01
    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ, SM = logpdf_GMM(X,gmm)
        llNew = SM.sum()/N
        P = np.exp(SJ-SM)
        gmmNew = []
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (gamma.reshape((1, gamma.size))*X).sum(1)
            S = np.dot(X, (gamma.reshape((1, gamma.size))*X).T)
            w = Z/N
            m = F/Z
            mu = m.reshape((m.size, 1))
            Sigma = S/Z - np.dot(mu, mu.T)
            U, s, _ = np.linalg.svd(Sigma)
            s[s<psi] = psi
            Sigma = np.dot(U, s.reshape((s.size, 1))*U.T)
            gmmNew.append((w, mu, Sigma))
        gmm = gmmNew
    
    return gmm

def GMM_EM_diag(X, gmm):
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]
    psi = 0.01
    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ, SM = logpdf_GMM(X,gmm)
        llNew = SM.sum()/N
        P = np.exp(SJ-SM)
        gmmNew = []
        
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (gamma.reshape((1, gamma.size))*X).sum(1)
            S = np.dot(X, (gamma.reshape((1, gamma.size))*X).T)
            w = Z/N
            m = F/Z
            mu = m.reshape((m.size, 1))
            Sigma = S/Z - np.dot(mu, mu.T)
            #diag
            Sigma = Sigma * np.eye(Sigma.shape[0])
            U, s, _ = np.linalg.svd(Sigma)
            s[s<psi] = psi
            sigma = np.dot(U, s.reshape((s.size, 1))*U.T)
            gmmNew.append((w, mu, sigma))
        gmm = gmmNew
        #print(llNew)
    #print(llNew-llOld)
    return gmm

def GMM_EM_tied(X, gmm):
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]
    #sigma_list = []
    psi = 0.01
    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ, SM = logpdf_GMM(X,gmm)
        llNew = SM.sum()/N
        P = np.exp(SJ-SM)
        gmmNew = []
        
        sigmaTied = np.zeros((X.shape[0],X.shape[0]))
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (gamma.reshape((1, gamma.size))*X).sum(1)
            S = np.dot(X, (gamma.reshape((1, gamma.size))*X).T)
            w = Z/N
            m = F/Z
            mu = m.reshape((m.size, 1))
            Sigma = S/Z - np.dot(mu, mu.T)
            sigmaTied += Z*Sigma
            gmmNew.append((w, mu))
        #get tied covariance
        gmm = gmmNew
        sigmaTied = sigmaTied/N
        U,s,_ = np.linalg.svd(sigmaTied)
        s[s<psi]=psi 
        sigmaTied = np.dot(U, s.reshape((s.size, 1))*U.T)
        
        gmmNew=[]
        for g in range(G):
            (w,mu)=gmm[g]
            gmmNew.append((w,mu,sigmaTied))
        gmm=gmmNew
        
        #print(llNew)
    #print(llNew-llOld)
    return gmm

def GMM_LBG(X, doub, version):
    assert version == 'full' or version == 'diagonal' or version == 'tied', "GMM version not correct"
    init_mu, init_sigma = mean_and_covarianceMatrix(X)
    gmm = [(1.0, init_mu, init_sigma)]
    
    for i in range(doub):
        doubled_gmm = []
        
        for component in gmm: 
            w = component[0]
            mu = component[1]
            sigma = component[2]
            U, s, Vh = np.linalg.svd(sigma)
            d = U[:, 0:1] * s[0]**0.5 * 0.1 # 0.1 is alpha
            component1 = (w/2, mu+d, sigma)
            component2 = (w/2, mu-d, sigma)
            doubled_gmm.append(component1)
            doubled_gmm.append(component2)
            if version == "full":
                gmm = GMM_EM(X, doubled_gmm)
            elif version == "diagonal":
                gmm = GMM_EM_diag(X, doubled_gmm)
            elif version == "tied":
                gmm = GMM_EM_tied(X, doubled_gmm)
            
    return gmm