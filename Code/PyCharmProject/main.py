import numpy as np
import matplotlib.pyplot as plt
import DataImport
import Constants as CONST
import DimensionalityReduction as DimRed

if __name__ == '__main__':
    DT, LT = DataImport.read_data('./Dataset/Train.txt')
    DT_male = DT[:, LT == 0]
    DT_female = DT[:, LT == 1]
    """
    for i in range(CONST.NFEATURES):
        plt.figure()
        plt.hist(DT_male[i, :], bins=10, alpha=0.4, ec='black')
        plt.hist(DT_female[i, :], bins=10, alpha=0.4, ec='black')
        plt.legend(['male', 'female'])
    # plt.show()
    """

    pca = DimRed.PCA(DT)
    DT_pca = pca.fitPCA(2)
    pca.scatter_2D_plot(DT_pca, LT)

    lda = DimRed.LDA(DT, LT)
    DT_lda = lda.fitLDA(2, True)
    lda.scatter_2D_plot(DT_lda, LT)

    plt.show()

