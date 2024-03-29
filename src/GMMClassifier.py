import numpy as np
import scipy.special


class GMM:
    def __init__(self, alpha=0.1, nComponents=2, psi=0.01, covType='Full'):
        self.alpha = alpha
        self.nComponents = nComponents
        self.psi = psi
        self.covType = covType

    def __logpdf_GAU_ND(self, X, mu, C):
        invC = np.linalg.inv(C)
        _, log_abs_detC = np.linalg.slogdet(C)
        M = X.shape[0]
        return - M/2 * np.log(2 * np.pi) - 0.5 * log_abs_detC - 0.5 * ((X - mu) * np.dot(invC, X - mu)).sum(0)

    def __logpdf_GMM(self, X, gmm):
        S = np.zeros((len(gmm), X.shape[1]))

        for g in range(len(gmm)):
            (w, mu, C) = gmm[g]
            S[g, :] = self.__logpdf_GAU_ND(X, mu, C) + np.log(w)

        logdens = scipy.special.logsumexp(S, axis=0)
        return S, logdens

    def __GMM_algorithm_EM(self, X, gmm, psi=0.01, cov='Full'):
        thNew = None
        thOld = None
        N = X.shape[1]
        D = X.shape[0]

        while thOld == None or thNew - thOld > 1e-6:
            thOld = thNew
            logSj, logSjMarg = self.__logpdf_GMM(X, gmm)
            thNew = np.sum(logSjMarg) / N

            P = np.exp(logSj - logSjMarg)

            if cov == 'Diag':
                newGmm = []
                for i in range(len(gmm)):
                    gamma = P[i, :]
                    Z = gamma.sum()
                    F = (gamma.reshape(1, -1) * X).sum(1)
                    S = np.dot(X, (gamma.reshape(1, -1) * X).T)
                    w = Z / N
                    mu = (F / Z).reshape(-1, 1)
                    sigma = S / Z - np.dot(mu, mu.T)
                    sigma *= np.eye(sigma.shape[0])
                    U, s, _ = np.linalg.svd(sigma)
                    s[s < psi] = psi
                    sigma = np.dot(U, s.reshape(-1, 1) * U.T)
                    newGmm.append((w, mu, sigma))
                gmm = newGmm

            elif cov == 'Tied':
                newGmm = []
                sigmaTied = np.zeros((D, D))
                for i in range(len(gmm)):
                    gamma = P[i, :]
                    Z = gamma.sum()
                    F = (gamma.reshape(1, -1) * X).sum(1)
                    S = np.dot(X, (gamma.reshape(1, -1) * X).T)
                    w = Z / N
                    mu = (F / Z).reshape(-1, 1)
                    sigma = S / Z - np.dot(mu, mu.T)
                    sigmaTied += Z * sigma
                    newGmm.append((w, mu))
                gmm = newGmm
                sigmaTied /= N
                U, s, _ = np.linalg.svd(sigmaTied)
                s[s < psi] = psi
                sigmaTied = np.dot(U, s.reshape(-1, 1) * U.T)

                newGmm = []
                for i in range(len(gmm)):
                    (w, mu) = gmm[i]
                    newGmm.append((w, mu, sigmaTied))

                gmm = newGmm

            elif cov == 'TiedDiag':
                newGmm = []
                sigmaTied = np.zeros((D, D))
                for i in range(len(gmm)):
                    gamma = P[i, :]
                    Z = gamma.sum()
                    F = (gamma.reshape(1, -1) * X).sum(1)
                    S = np.dot(X, (gamma.reshape(1, -1) * X).T)
                    w = Z / N
                    mu = (F / Z).reshape(-1, 1)
                    sigma = S / Z - np.dot(mu, mu.T)
                    sigmaTied += Z * sigma
                    newGmm.append((w, mu))
                gmm = newGmm
                sigmaTied /= N
                sigmaTied *= np.eye(sigma.shape[0])
                U, s, _ = np.linalg.svd(sigmaTied)
                s[s < psi] = psi
                sigmaTied = np.dot(U, s.reshape(-1, 1) * U.T)

                newGmm = []
                for i in range(len(gmm)):
                    (w, mu) = gmm[i]
                    newGmm.append((w, mu, sigmaTied))

                gmm = newGmm

            else:
                newGmm = []
                for i in range(len(gmm)):
                    gamma = P[i, :]
                    Z = gamma.sum()
                    F = (gamma.reshape(1, -1) * X).sum(1)
                    S = np.dot(X, (gamma.reshape(1, -1) * X).T)
                    w = Z / N
                    mu = (F / Z).reshape(-1, 1)
                    sigma = S / Z - np.dot(mu, mu.T)
                    U, s, _ = np.linalg.svd(sigma)
                    s[s < psi] = psi
                    sigma = np.dot(U, s.reshape(-1, 1) * U.T)
                    newGmm.append((w, mu, sigma))
                gmm = newGmm
        return gmm

    def __GMM_algorithm_LBG(self, X, alpha, nComponents, psi=0.01, covType='Full'):
        mean = X.mean(axis=1).reshape(-1, 1)
        cov = 1 / X.shape[1] * np.dot(X - mean, (X - mean).T)
        gmm = [(1, mean, cov)]

        while len(gmm) <= nComponents:
            gmm = self.__GMM_algorithm_EM(X, gmm, psi, covType)

            if len(gmm) == nComponents:
                break

            newGmm = []
            for i in range(len(gmm)):
                (w, mu, sigma) = gmm[i]
                U, s, Vh = np.linalg.svd(sigma)
                d = U[:, 0:1] * s[0] ** 0.5 * alpha
                newGmm.append((w / 2, mu + d, sigma))
                newGmm.append((w / 2, mu - d, sigma))
            gmm = newGmm

        return gmm

    def train(self, Dtrain, Ltrain):
        self.Dtrain_c0 = Dtrain[:, Ltrain == 0]
        self.Dtrain_c1 = Dtrain[:, Ltrain == 1]
        self.gmm_c0 = self.__GMM_algorithm_LBG(self.Dtrain_c0, self.alpha, self.nComponents, self.psi, self.covType)
        self.gmm_c1 = self.__GMM_algorithm_LBG(self.Dtrain_c1, self.alpha, self.nComponents, self.psi, self.covType)
        return self

    def predict(self, Dtest, labels=False):
        _, llr_0 = self.__logpdf_GMM(Dtest, self.gmm_c0)
        _, llr_1 = self.__logpdf_GMM(Dtest, self.gmm_c1)
        if labels:
            S = np.vstack([llr_0.reshape(1, -1), llr_1.reshape(1, -1)])
            return np.argmax(S, axis=0)
        else:
            return llr_1 - llr_0