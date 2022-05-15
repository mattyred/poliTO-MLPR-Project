import numpy as np
class DimRed:
    def __init__(self, D, L):
        self.D = D
        self.L = L
        self.N = D.shape[1]
    
    def __computeProjectionMatrix(self, ndim):
        mu = self.D.mean(axis = 1).reshape(-1,1)
        Dc = self.D - mu
        C = 1/self.N * np.dot(Dc, Dc.T) # covariance matrix
        sigma, U = np.linalg.eigh(C)
        P = U[:, ::-1][:, 0:ndim] # take the m eigenvectos of C associated to the m highest eigenvalues
        return P
    
    def PCADataset(self, ndim):
        P = self.__computeProjectionMatrix(ndim)
        Dprojected = np.dot(P.T, self.D)
        return Dprojected
    
    def mostRelevantFeatures(self):
        
    def LDADataset(self, ndim):
        pass