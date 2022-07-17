import Constants
import numpy as np
from scipy.stats import norm
import seaborn
import matplotlib.pyplot as plt
class DataPreProcesser:

    @staticmethod
    def gaussianized_features_training(DT):
        F = DT.shape[0]
        N = DT.shape[1]
        ranks = []
        for j in range(F):
            rank_num = 0
            for i in range(N):
                rank_num += (DT[j, :] > DT[j, i]).astype(int)
            rank_num += 1
            ranks.append(rank_num / (N+2))
        y = norm.ppf(ranks)
        return y

    @staticmethod
    def gaussianized_features_evaluation(DE, DT):
        F = DT.shape[0]
        NT = DT.shape[1]
        NE = DE.shape[1]
        ranks = []
        for j in range(F):
            rank_num = 0
            for i in range(NT):
                rank_num += (DE[j, :] > DT[j, i]).astype(int)
            rank_num += 1
            ranks.append(rank_num / (NT + 2))
        y = norm.ppf(ranks)
        return y

    @staticmethod
    def znormalized_features_training(DT):
        DTmean = DT.mean(axis=1).reshape(-1, 1)
        DTstdDev = DT.std(axis=1).reshape(-1, 1)
        ZnormDT = (DT - DTmean) / DTstdDev
        return ZnormDT

    @staticmethod
    def znormalized_features_evaluation(DE, DT):
        DTmean = DT.mean(axis=1).reshape(-1, 1)
        DTstdDev = DT.std(axis=1).reshape(-1, 1)
        ZnormDE = (DE - DTmean) / DTstdDev
        return ZnormDE

    @staticmethod
    def heatmap(DT, LT, plt, title):
        fig, axs = plt.subplots(1, 3)
        fig.suptitle(title)
        seaborn.heatmap(np.corrcoef(DT), linewidth=0.2, cmap="Greys", square=True, cbar=False, ax=axs[0])
        seaborn.heatmap(np.corrcoef(DT[:, LT == 0]), linewidth=0.2, cmap="Reds", square=True, cbar=False, ax=axs[1])
        seaborn.heatmap(np.corrcoef(DT[:, LT == 1]), linewidth=0.2, cmap="Blues", square=True, cbar=False, ax=axs[2])

