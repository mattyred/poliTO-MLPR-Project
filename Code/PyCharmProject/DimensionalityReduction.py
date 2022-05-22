import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


class PCA:
    def __init__(self, D):
        self.D = D
        self.N = D.shape[1]

    def __computeProjectionMatrix(self, ndim):
        mu = self.D.mean(axis=1).reshape(-1, 1)
        Dc = self.D - mu
        C = 1 / self.N * np.dot(Dc, Dc.T)  # covariance matrix
        sigma, U = np.linalg.eigh(C)
        P = U[:, ::-1][:, 0:ndim]  # take the m eigenvectos of C associated to the m highest eigenvalues
        return P

    def fitPCA(self, ndim):
        P = self.__computeProjectionMatrix(ndim)
        Dprojected = np.dot(P.T, self.D)
        return Dprojected

    def scatter_2D_plot(self, DT_pca, LT):
        plt.figure()
        plt.scatter(DT_pca[0, LT == 0], DT_pca[1, LT == 0])
        plt.scatter(DT_pca[0, LT == 1], DT_pca[1, LT == 1])

class LDA:
    def __init__(self, D, L):
        self.D = D
        self.L = L
        self.N = D.shape[1]
        self.F = D.shape[0]
        self.K = len(set(L))
        self.mu = np.mean(D, axis=1).reshape(-1, 1)
        self.nc = np.array([np.sum(L == i) for i in set(L)])

    def __computeSB(self):
        Sb = 0
        mean_classes = self.__computeMC() - self.mu
        for i in range(self.K):
            Sb += self.nc[i] * np.dot(mean_classes[:, i:i + 1], mean_classes[:, i:i + 1].T)
        Sb /= sum(self.nc)
        return Sb

    def __computeSW(self):
        Swc = 0
        Sw = 0
        for c in range(self.K):
            Dc = self.D[:, self.L == c]
            Dc -= np.mean(Dc, axis=1).reshape(-1, 1)
            Swc = 1 / self.nc[c] * np.dot(Dc, Dc.T)
            Sw += self.nc[c] * Swc
        Sw /= sum(self.nc)
        return Sw

    def __computeMC(self):
        mean_classes = np.zeros(shape=(self.F, self.K))
        for c in range(self.K):
            Dc = self.D[:, self.L == c]
            Mc = np.mean(Dc, axis=1).reshape(-1, 1)
            mean_classes[:, c:c + 1] = Mc
        return mean_classes

    def fitLDA(self, ndim, ortho=False):
        Sb = self.__computeSB()
        Sw = self.__computeSW()
        s, U = scipy.linalg.eigh(Sb, Sw)
        W = U[:, ::-1][:, 0:ndim]
        if ortho:
            UW, _, _ = np.linalg.svd(W)
            U = UW[:, 0:ndim]
            Dprojected = np.dot(U.T, self.D)
        else:
            Dprojected = np.dot(W.T, self.D)
        return Dprojected

    def scatter_2D_plot(self, DT_lda, LT):
        plt.figure()
        plt.scatter(DT_lda[0, LT == 0], DT_lda[1, LT == 0])
        plt.scatter(DT_lda[0, LT == 1], DT_lda[1, LT == 1])
