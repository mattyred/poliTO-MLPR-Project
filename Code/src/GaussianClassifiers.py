import numpy as np
import scipy
import matplotlib.pyplot as plt


class MVG:
    """
        F: # features
        N: # samples
        K: # classes
        Dtrain: shape [F, N]
        Ltrain: shape [N,]
    """

    def __init__(self):
        self.mu_classes = []  # list of empiracal means for each class
        self.cov_classes = []  # list of covariance matrices for each class

    """
    The training phase consists in computing empirical mean and covariance matrix for rach class
    These are the only model parameters we need to fit the model to a new (test) dataset
    """
    def train(self, Dtrain, Ltrain):
        self.Dtrain = Dtrain
        self.Ltrain = Ltrain
        self.N = Dtrain.shape[1]
        self.F = Dtrain.shape[0]
        self.K = len(set(Ltrain))
        self.labels = set(self.Ltrain)
        self.mu_classes = []
        self.cov_classes = []
        for i in self.labels:
            Dtrain_i = self.Dtrain[:, self.Ltrain == i]
            N_i = Dtrain_i.shape[1]
            mu_i = Dtrain_i.mean(axis=1).reshape(-1, 1)
            cov_i = 1 / N_i * np.dot(Dtrain_i - mu_i, (Dtrain_i - mu_i).T)
            self.mu_classes.append(mu_i)
            self.cov_classes.append(cov_i)
        return self

    def __logpdf_GAU_ND_1sample(self, x, mu, C):
        M = x.shape[0]
        mu = mu.reshape(M, 1)
        xc = x - mu
        invC = np.linalg.inv(C)
        _, log_abs_detC = np.linalg.slogdet(C)
        return -M / 2 * np.log(2 * np.pi) - 1 / 2 * log_abs_detC - 1 / 2 * np.dot(np.dot(xc.T, invC), xc)

    def predict(self, Dtest, labels=True):
        Ntest = Dtest.shape[1] # number of test samples
        S = np.zeros(shape=(self.K, Ntest))
        for i in range(Ntest):
            xt = Dtest[:, i:i + 1]
            score = np.zeros(shape=(self.K, 1))
            for j in range(self.K):
                mu = self.mu_classes[j]
                C = self.cov_classes[j]
                score[j, :] = np.exp(self.__logpdf_GAU_ND_1sample(xt, mu, C))
            S[:, i:i + 1] = score
        if labels:
            SJoint = 1 / 2 * S
            logSJoint = np.log(SJoint) + np.log(1 / 2)
            logSMarginal = scipy.special.logsumexp(logSJoint, axis=0).reshape(1, -1)
            log_SPost = logSJoint - logSMarginal
            SPost = np.exp(log_SPost)
            predicted_labels = np.argmax(SPost, axis=0)
            return predicted_labels
        else:
            return np.log(S[1, :]) - np.log(S[0, :])
        ## np.exp(self.Spost[:, 1]) / (np.exp(self.Spost[:, 0]) + 0.00000001) => What we have to return


class TiedG:
    """
            F: # features
            N: # samples
            K: # classes
            Dtrain: shape [F, N]
            Ltrain: shape [N,]
    """

    def __init__(self):
        pass

    """
    The training phase consists in computing empirical mean for each class and only one covarince matrix for all classes
    """
    def train(self, Dtrain, Ltrain):
        self.Dtrain = Dtrain
        self.Ltrain = Ltrain
        self.labels = set(self.Ltrain)
        self.N = Dtrain.shape[1]
        self.F = Dtrain.shape[0]
        self.K = len(set(Ltrain))
        cov_classes = []
        self.mu_classes = []  # list of empiracal means for each class
        self.tied_cov = 0 # tied covariance matrix
        for i in self.labels:
            Dtrain_i = self.Dtrain[:, self.Ltrain == i]
            N_i = Dtrain_i.shape[1]
            mu_i = Dtrain_i.mean(axis=1).reshape(-1, 1)
            cov_i = 1 / N_i * np.dot(Dtrain_i - mu_i, (Dtrain_i - mu_i).T)
            self.mu_classes.append(mu_i)
            cov_classes.append(cov_i)
        num_samples_per_class = [sum(self.Ltrain == i) for i in range(self.K)]
        for i in range(self.K):
            self.tied_cov += (num_samples_per_class[i] * cov_classes[i])
        self.tied_cov *= 1 / sum(num_samples_per_class)
        return self

    def __logpdf_GAU_ND_1sample(self, x, mu, C):
        M = x.shape[0]
        mu = mu.reshape(M, 1)
        xc = x - mu
        invC = np.linalg.inv(C)
        _, log_abs_detC = np.linalg.slogdet(C)
        return -M / 2 * np.log(2 * np.pi) - 1 / 2 * log_abs_detC - 1 / 2 * np.dot(np.dot(xc.T, invC), xc)

    def predict(self, Dtest, labels=True):
        Ntest = Dtest.shape[1]
        S = np.zeros(shape=(self.K, Ntest))
        for i in range(Ntest):
            xt = Dtest[:, i:i + 1]
            score = np.zeros(shape=(self.K, 1))
            for j in range(self.K):
                mu = self.mu_classes[j]
                C = self.tied_cov
                score[j, :] = np.exp(self.__logpdf_GAU_ND_1sample(xt, mu, C))
            S[:, i:i + 1] = score
        if labels:
            SJoint = 1 / 3 * S
            SMarginal = SJoint.sum(0).reshape(-1, 1)
            SPost = np.zeros(shape=(self.K, Ntest))
            for c in range(self.K):
                SJoint_c = SJoint[c, :].reshape(-1, 1)
                SPost_c = (SJoint_c / SMarginal).reshape(1, -1)
                SPost[c, :] = SPost_c
            predicted_labels = np.argmax(SPost, axis=0)
            return predicted_labels
        else:
            return np.log(S[1, :]) - np.log(S[0, :])

    def get_decision_function_parameters(self):
        precision_matrix = np.linalg.inv(self.tied_cov)
        b = np.dot(precision_matrix, (self.mu_classes[1] - self.mu_classes[0]))
        c = -1/2 * (np.dot(np.dot(self.mu_classes[1].T, precision_matrix), self.mu_classes[1])-
                    np.dot(np.dot(self.mu_classes[0].T, precision_matrix), self.mu_classes[0]))
        return b, c


class NaiveBayes:
    """
                F: # features
                N: # samples
                K: # classes
                Dtrain: shape [F, N]
                Ltrain: shape [N,]
            """

    def __init__(self):
        pass

    def train(self,  Dtrain, Ltrain):
        self.Dtrain = Dtrain
        self.Ltrain = Ltrain
        self.labels = set(self.Ltrain)
        self.N = Dtrain.shape[1]
        self.F = Dtrain.shape[0]
        self.K = len(set(Ltrain))
        self.mu_classes = []  # list of empiracal means for each class
        self.diag_cov_classes = []  # diagonal covariance matrices for each class
        cov_classes = []
        for i in self.labels:
            Dtrain_i = self.Dtrain[:, self.Ltrain == i]
            N_i = Dtrain_i.shape[1]
            mu_i = Dtrain_i.mean(axis=1).reshape(-1, 1)
            cov_i = 1 / N_i * np.dot(Dtrain_i - mu_i, (Dtrain_i - mu_i).T)
            self.mu_classes.append(mu_i)
            cov_classes.append(cov_i)

        for i in range(self.K):
            self.diag_cov_classes.append(cov_classes[i] * np.identity(self.F))
        return self

    def __logpdf_GAU_ND_1sample(self, x, mu, C):
        M = x.shape[0]
        mu = mu.reshape(M, 1)
        xc = x - mu
        invC = np.linalg.inv(C)
        _, log_abs_detC = np.linalg.slogdet(C)
        return -M / 2 * np.log(2 * np.pi) - 1 / 2 * log_abs_detC - 1 / 2 * np.dot(np.dot(xc.T, invC), xc)

    def predict(self, Dtest, labels=True):
        Ntest = Dtest.shape[1]
        S = np.zeros(shape=(self.K, Ntest))
        for i in range(Ntest):
            xt = Dtest[:, i:i + 1]
            score = np.zeros(shape=(self.K, 1))
            for j in range(self.K):
                mu = self.mu_classes[j]
                C = self.diag_cov_classes[j]
                score[j, :] = np.exp(self.__logpdf_GAU_ND_1sample(xt, mu, C))
            S[:, i:i + 1] = score
        if labels:
            SJoint = 1 / 3 * S
            SMarginal = SJoint.sum(0).reshape(-1, 1)
            SPost = np.zeros(shape=(self.K, Ntest))
            for c in range(self.K):
                SJoint_c = SJoint[c, :].reshape(-1, 1)
                SPost_c = (SJoint_c / SMarginal).reshape(1, -1)
                SPost[c, :] = SPost_c
            predicted_labels = np.argmax(SPost, axis=0)
        else:
            return np.log(S[1, :]) - np.log(S[0, :])