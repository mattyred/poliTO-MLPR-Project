import numpy as np
import scipy.special
import scipy.optimize

class LinearLogisticRegression:

    def __init__(self, lbd, factr):
        self.lbd = lbd
        self.factr = factr


    def __compute_zi(self, ci):
        return 2 * ci - 1

    def __logreg_obj(self, v):  # still works if DTR is one sample only? yes but it must be of shape (4,1)
        w, b = v[0:-1], v[-1]
        J = self.lbd / 2 * (np.linalg.norm(w) ** 2)
        summary = 0
        for i in range(self.N):
            xi = self.Dtrain[:, i:i + 1]
            ci = self.Ltrain[i]
            zi = self.__compute_zi(ci)
            summary += np.logaddexp(0, -zi * (np.dot(w.T, xi) + b))
        J += (1 / self.N) * summary
        return J

    def train(self, Dtrain, Ltrain):
        self.Dtrain = Dtrain
        self.Ltrain = Ltrain
        self.F = Dtrain.shape[0]  # dimensionality of features space
        self.K = len(set(Ltrain))  # number of classes
        self.N = Dtrain.shape[1]
        self.x, f, d = scipy.optimize.fmin_l_bfgs_b(func=self.__logreg_obj,
                                                    x0=np.zeros(self.Dtrain.shape[0] + 1),
                                                    approx_grad=True,
                                                    iprint=0,
                                                    factr=self.factr)

        """                                            
        print('Point of minimum: %s' % (self.x))
        print('Value of the minimum: %s' % (f))
        print('Number of iterations: %s' % (d['funcalls']))
        """
        return self

    def predict(self, Dtest, labels=True):
        w, b = self.x[0:-1], self.x[-1]
        S = np.zeros((Dtest.shape[1]))
        for i in range(Dtest.shape[1]):
            xi = Dtest[:, i:i + 1]
            s = np.dot(w.T, xi) + b
            S[i] = s
        if labels:
            LP = S > 0
            return LP
        else:
            return S


class QuadraticLogisticRegression:
    def __init__(self, lbd, factr):
        self.lbd = lbd
        self.factr = factr

    def __compute_zi(self, ci):
        return 2 * ci - 1

    def __logreg_obj(self, v):  # still works if DTR is one sample only? yes but it must be of shape (4,1)
        w, b = v[0:-1], v[-1]
        J = self.lbd / 2 * (np.linalg.norm(w) ** 2)
        summary = 0
        for i in range(self.N):
            xi = self.DTR_ext[:, i:i + 1]
            ci = self.Ltrain[i]
            zi = self.__compute_zi(ci)
            summary += np.logaddexp(0, -zi * (np.dot(w.T, xi) + b))
        J += (1 / self.N) * summary
        return J

    def train(self, Dtrain, Ltrain):
        self.Dtrain = Dtrain
        self.Ltrain = Ltrain
        self.F = Dtrain.shape[0]  # dimensionality of features space
        self.K = len(set(Ltrain))  # number of classes
        self.N = Dtrain.shape[1]
        self.DTR_ext = np.hstack([self.__expandFeatures(self.Dtrain[:, i]) for i in range(self.N)])
        self.x, f, d = scipy.optimize.fmin_l_bfgs_b(func=self.__logreg_obj,
                                                    x0=np.zeros(self.DTR_ext.shape[0] + 1),
                                                    approx_grad=True,
                                                    iprint=0,
                                                    factr=self.factr,
                                                    maxiter=5)
        return self

    def __expandFeatures(self, x):
        x = x.reshape(-1, 1)
        expanded_x = np.dot(x, x.T).reshape(-1, 1)
        return np.vstack([expanded_x, x])

    def predict(self, Dtest, labels=False):
        DTE_ext = np.hstack([self.__expandFeatures(Dtest[:, i]) for i in range(Dtest.shape[1])])
        w, b = self.x[0:-1], self.x[-1]
        scores = np.dot(w.T, DTE_ext) + b
        if labels:
            LP = scores > 0
            return LP
        else:
            return scores