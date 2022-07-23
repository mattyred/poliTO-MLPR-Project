import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
from itertools import repeat

class SVM:
    def __init__(self, hparams, kernel=None):
        self.kernelType = kernel
        self.C = hparams['C']
        self.K = hparams['K']
        self.eps = hparams.get('eps')
        self.gamma = hparams.get('gamma')
        self.c = hparams.get('c')
        self.d = hparams.get('d')

    def __LDc_obj(self, alpha):
        #grad = np.dot(self.H, alpha) - np.ones(self.H.shape[1])
        t = 0.5 * np.dot(np.dot(alpha.reshape(1, -1), self.H), alpha) - alpha.sum(), np.dot(self.H, alpha) - 1
        return t

    def __polynomial_kernel(self, X1, X2):
        ker = (np.dot(X1.T, X2) + self.c) ** self.d + self.K ** 2
        return ker

    def __RBF_kernel(self, X1, X2):
        x = np.repeat(X1, X2.shape[1], axis=1)
        y = np.tile(X2, X1.shape[1])
        ker = np.exp(
            -self.gamma * np.linalg.norm(x - y, axis=0).reshape(X1.shape[1], X2.shape[1]) ** 2) + self.K ** 2
        return ker

    def train(self, Dtrain, Ltrain):
        self.Dtrain = Dtrain
        self.Ltrain = Ltrain
        self.N = Dtrain.shape[1]
        self.Ltrain_z = self.Ltrain * 2 - 1
        self.Ltrain_z_matrix = self.Ltrain_z.reshape(-1, 1) * self.Ltrain_z.reshape(1, -1)
        self.bounds = np.array([(0, self.C)] * Ltrain.shape[0])

        if self.kernelType is not None:
            if self.kernelType == 'Polynomial':
                ker = self.__polynomial_kernel(self.Dtrain, self.Dtrain)
            elif self.kernelType == 'RBF':
                ker = self.__RBF_kernel(self.Dtrain, self.Dtrain)
            else:
                return
            self.H = self.Ltrain_z_matrix * ker
        else:
            # Compute expanded D matrix
            self.expandedD = np.vstack((Dtrain, self.K * np.ones(self.N)))
            # Compute H matrix
            G = np.dot(self.expandedD.T, self.expandedD)
            self.H = G * self.Ltrain_z_matrix

        self.alpha, self.primal, _ = scipy.optimize.fmin_l_bfgs_b(func=self.__LDc_obj,
                                                                    bounds=self.bounds,
                                                                    x0=np.zeros(Dtrain.shape[1]),
                                                                    factr=1.0)

        return self

    def predict(self, Dtest, labels=False):
        if self.kernelType is not None:
            if self.kernelType == 'Polynomial':
                self.S = np.sum(np.dot((self.alpha * self.Ltrain_z).reshape(1, -1), self.__polynomial_kernel(self.Dtrain, Dtest)), axis=0)
            elif self.kernelType == 'RBF':
                self.S = np.sum(np.dot((self.alpha * self.Ltrain_z).reshape(1, -1), self.__RBF_kernel(self.Dtrain, Dtest)), axis=0)
            else:
                return
        else:
            wc = np.sum(self.alpha * self.Ltrain_z * self.expandedD, axis=1)
            self.w, self.b = wc[:-1], wc[-1::]
            self.S = np.dot(self.w.T, Dtest) + self.b * self.K

        if labels is True:
            predicted_labels = np.where(self.S > 0, 1, 0)
            return predicted_labels
        else:
            return self.S

    """
    def plot_decision_2D_hyperplane(self, X2D, LTE):
        min_feature_x = min(X2D[0, :])
        max_feature_x = max(X2D[0, :])
        min_feature_y = min(X2D[1, :])
        max_feature_y = max(X2D[1, :])
        x_points = np.linspace(min_feature_x-0.5, max_feature_x+0.5)
        y_points = -(self.w[0] / self.w[1]) * x_points - self.b / self.w[1]
        plt.figure(figsize=(10, 8))
        plt.xlim(min_feature_x-0.5, max_feature_x+0.5)
        plt.ylim(min_feature_y-0.5, max_feature_y+0.5)
        colormap = {0: 'b', 1: 'g'}
        for label in range(2):
            plt.scatter(X2D[0, LTE == label], X2D[1, LTE == label], color=colormap[label])
        self.__fill_regions(X2D, LTE, x_points, y_points, min_feature_y, max_feature_y, colormap)
        plt.plot(x_points, y_points, c='r', ls='--')
        plt.show()
        
    def __fill_regions(self, X2D, LTE, x_points, y_points, miny, maxy, colormap):
        test_point = np.array([X2D[0, 0], X2D[1, 0]]).reshape(-1, 1)
        test_point_label = LTE[0]
        if test_point[1] < -(self.w[0] / self.w[1]) * test_point[0] - self.b / self.w[1]:
            # point below line
            color_below = colormap[test_point_label]
            color_above = colormap[0] if test_point_label == 1 else colormap[1]
        else:  # point above line
            color_above = colormap[test_point_label]
            color_below = colormap[0] if test_point_label == 1 else colormap[1]
        plt.fill_between(x_points, y_points, miny - 0.5, color=color_below, alpha=0.2)
        plt.fill_between(x_points, y_points, maxy + 0.5, color=color_above, alpha=0.2)
    """
