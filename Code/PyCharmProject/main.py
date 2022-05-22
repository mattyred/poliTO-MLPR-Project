import numpy as np
import matplotlib.pyplot as plt
import DataImport
import Constants as CONST
import DimensionalityReduction as DimRed
import GaussianClassifiers as GauClf
import ModelEvaluation

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0) # 2/3 of the dataset D are used for training, 1/3 for validation
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

if __name__ == '__main__':
    DT, LT = DataImport.read_data('./Dataset/Train.txt')
    (DTR, LTR), (DTE, LTE) = split_db_2to1(DT, LT)
    DTR_male = DTR[:, LTR == 0]
    DTR_female = DTR[:, LTR == 1]
    """
    for i in range(CONST.NFEATURES):
        plt.figure()
        plt.hist(DT_male[i, :], bins=10, alpha=0.4, ec='black')
        plt.hist(DT_female[i, :], bins=10, alpha=0.4, ec='black')
        plt.legend(['male', 'female'])
    # plt.show()
    """

    # --- Dimensionality Reduction ---
    pca = DimRed.PCA(DTR)
    DTR_pca = pca.fitPCA(2)
    pca.scatter_2D_plot(DTR_pca, LTR)

    lda = DimRed.LDA(DTR, LTR)
    DTR_lda = lda.fitLDA(2, True)
    lda.scatter_2D_plot(DTR_lda, LTR)

    # plt.show()

    # --- Gaussian Models ---
    model_evaluator = ModelEvaluation.BinaryModelEvaluator()

    # MVG
    mvg_clf = GauClf.MVG(DTR, LTR)
    labels = mvg_clf.train().predict(DTE, labels=True)
    M = model_evaluator.confusion_matrix(labels, LTE)
    print('MVG error rate: %.2f%%' % (model_evaluator.error_rate(labels, LTE) * 100))

    # Tied Covariance
    tied_clf = GauClf.TiedG(DTR, LTR)
    tied_clf.train()
    labels = tied_clf.train().predict(DTE, labels=True)
    M = model_evaluator.confusion_matrix(labels, LTE)
    print('Tied Covariance error rate: %.2f%%' % (model_evaluator.error_rate(labels, LTE) * 100))

    # Naive Bayes
    naiveBayes_clf = GauClf.NaiveBayes(DTR, LTR)
    naiveBayes_clf.train()
    labels = naiveBayes_clf.train().predict(DTE, labels=True)
    M = model_evaluator.confusion_matrix(labels, LTE)
    print('Naive Bayes error rate: %.2f%%' % (model_evaluator.error_rate(labels, LTE) * 100))



