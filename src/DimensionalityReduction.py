import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


class PCA:
    def __init__(self, D=None):
        if D is not None:
            self.D = D
            self.N = D.shape[1]
        else:
            pass

    def __computeProjectionMatrix(self, ndim):
        mu = self.D.mean(axis=1).reshape(-1, 1)
        Dc = self.D - mu
        C = 1 / self.N * np.dot(Dc, Dc.T)  # covariance matrix
        sigma, U = np.linalg.eigh(C)
        P = U[:, ::-1][:, 0:ndim]  # take the m eigenvectos of C associated to the m highest eigenvalues
        return P

    def fitPCA_train(self, ndim):
        self.P = self.__computeProjectionMatrix(ndim)
        Dprojected = np.dot(self.P.T, self.D)
        return Dprojected

    def fitPCA_test(self, Dtest):
        Dtestprojected = np.dot(self.P.T, Dtest)
        return Dtestprojected

    def scatter_2D_plot(self, DT_pca, LT):
        plt.figure()
        plt.scatter(DT_pca[0, LT == 0], DT_pca[1, LT == 0])
        plt.scatter(DT_pca[0, LT == 1], DT_pca[1, LT == 1])

    @staticmethod
    def computeSigmaMatrix(Dtrain):
        mu = Dtrain.mean(axis=1).reshape(-1, 1)
        Dc = Dtrain - mu
        C = 1 / Dtrain.shape[1] * np.dot(Dc, Dc.T)  # covariance matrix
        sigma, _ = np.linalg.eigh(C)
        return sigma

    @staticmethod
    def kfold_PCA(D=None, k=3, threshold=0.95, show=False):
        Nsamples = D.shape[1]
        np.random.seed(0)
        idx = np.random.permutation(Nsamples)
        folds = np.array_split(idx, k)
        m_values = np.arange(1, D.shape[0])[::-1]
        avg_perc_values = []
        for m in m_values:
            # For each value of m in 1..11 use k-fold cross validation to know if, using that m, PCA
            # is able to extract features with variance > threshold
            avg_perc = 0
            for i in range(k):
                fold_test = folds[i]
                folds_train = []
                for j in range(k):
                    if j != i:
                        folds_train.append(folds[j])

                Dtrain = D[:, np.array(folds_train).flat]
                sigma = PCA.computeSigmaMatrix(Dtrain)
                largest_eigh = np.flip(sigma)[0:m]
                t = sum(largest_eigh) / sum(sigma)
                avg_perc += t
            avg_perc /= k
            avg_perc_values.append(avg_perc)
        if show:
            plt.figure()
            plt.plot(m_values, np.array(avg_perc_values) * 100, '--o')
            plt.xticks(m_values)
            plt.xlabel('m')
            plt.ylabel('% of retained variance of data')
            plt.show()
        return m_values, np.array(avg_perc_values)

