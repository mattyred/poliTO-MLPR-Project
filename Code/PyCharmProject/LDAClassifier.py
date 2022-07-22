import numpy as np
import scipy.linalg

class LDA:
    def __init__(self):
        pass

    def train(self, Dtrain, Ltrain):
        self.Dtrain = Dtrain
        self.Ltrain = Ltrain
        self.N = Dtrain.shape[1]
        self.F = Dtrain.shape[0]
        self.K = len(set(Ltrain))
        self.mu = np.mean(Dtrain, axis=1).reshape(-1, 1)
        self.nc = np.array([np.sum(Ltrain == i) for i in set(Ltrain)])
        return self

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
            Dc = self.Dtrain[:, self.Ltrain == c]
            Dc -= np.mean(Dc, axis=1).reshape(-1, 1)
            Swc = 1 / self.nc[c] * np.dot(Dc, Dc.T)
            Sw += self.nc[c] * Swc
        Sw /= sum(self.nc)
        return Sw

    def __computeMC(self):
        mean_classes = np.zeros(shape=(self.F, self.K))
        for c in range(self.K):
            Dc = self.Dtrain[:, self.Ltrain == c]
            Mc = np.mean(Dc, axis=1).reshape(-1, 1)
            mean_classes[:, c:c + 1] = Mc
        return mean_classes

    def predict(self, Dtest, labels=False):
        ndim = 1
        Sb = self.__computeSB()
        Sw = self.__computeSW()
        s, U = scipy.linalg.eigh(Sb, Sw)
        W = U[:, ::-1][:, 0:ndim]
        scores = np.dot(W.T, Dtest)
        if labels:
            return scores > 0
        return scores.reshape(Dtest.shape[1],)

    def get_decision_function_parameters(self):
        precision_matrix = np.linalg.inv(self.__computeSW())
        mu_classes = self.__computeMC()
        b = np.dot(precision_matrix, (mu_classes[:, 1] - mu_classes[:, 0]))
        return b

    """
        def scatter_2D_plot(self, DT_lda, LT):
        plt.figure()
        plt.scatter(DT_lda[0, LT == 0], DT_lda[1, LT == 0])
        plt.scatter(DT_lda[0, LT == 1], DT_lda[1, LT == 1])
    """

