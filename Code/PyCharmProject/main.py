import numpy as np
import matplotlib.pyplot as plt
import DataImport
import Constants as CONST
import DimensionalityReduction as DimRed
import GaussianClassifiers as GauClf
import ModelEvaluation
import PreProcessing
if __name__ == '__main__':
    DT, LT = DataImport.read_data_training('./Dataset/Train.txt')
    DE, LE = DataImport.read_data_evaluation('./Dataset/Test.txt')
    """
    # Features analysis - overall statistics
    m = np.mean(DT, axis=1).reshape(-1, 1)
    std = np.std(DT, axis=1).reshape(-1, 1)

    # Features analysis - no pre processing
    fig1, axs1 = plt.subplots(3, 4)
    fig1.suptitle('No preprocessing')
    for i in range(CONST.NFEATURES):
        axs1[i % 3, i % 4].hist(DT[i, LT == 0], bins=10, alpha=0.6, ec='black', density=True)  # male
        axs1[i % 3, i % 4].hist(DT[i, LT == 1], bins=10, alpha=0.6, ec='black', density=True)   # female

    # Features analysis - gaussianization pre processing
    PreProcesser = PreProcessing.DataPreProcesser()
    DTgaussianized = PreProcesser.gaussianized_features_training(DT)
    fig2, axs2 = plt.subplots(3, 4)
    fig2.suptitle('Gaussianization preprocessing')
    for i in range(CONST.NFEATURES):
        axs2[i % 3, i % 4].hist(DTgaussianized[i, LT == 0], bins=10, alpha=0.6, ec='black', density=True)  # male
        axs2[i % 3, i % 4].hist(DTgaussianized[i, LT == 1], bins=10, alpha=0.6, ec='black', density=True)   # female

    # Features analysis - znormalization pre processing
    DTznorm = PreProcesser.znormalized_features_training(DT)
    fig3, axs3 = plt.subplots(3, 4)
    fig3.suptitle('Z-normaliazation preprocessing')
    for i in range(CONST.NFEATURES):
        axs3[i % 3, i % 4].hist(DTznorm[i, LT == 0], bins=10, alpha=0.6, ec='black', density=True)  # male
        axs3[i % 3, i % 4].hist(DTznorm[i, LT == 1], bins=10, alpha=0.6, ec='black', density=True)   # female

    # Features analysis - correlation of non-preprocessed features
    PreProcesser.heatmap(DT, LT, plt, 'Features correlation (no preprocessing)')
    # Features analyssis - correlation of gaussianized features
    PreProcesser.heatmap(DTgaussianized, LT, plt, 'Features correlation (gaussianized features)')
    # Features analyssis - correlation of gaussianized features
    PreProcesser.heatmap(DTznorm, LT, plt, 'Features correlation (z-normalized features)')
    plt.show()
    """

    # Test k-fold cross validation (on MVG classifier)
    model_evaluator = ModelEvaluation.BinaryModelEvaluator()
    pi = 0.3
    Cfn = 1
    Cfp = 1
    selected_app = {'pi': pi, 'Cfn': Cfn, 'Cfp': Cfp}
    dim_red = None#{'type': 'pca', 'm': 9}

    print('R: MVG Classifier\nPreprocessing: -\nDim. Reduction: %s\nApplication: (pi=%.2f, Cfn=%.2f, Cfp=%.2f)\nValidation: k-fold' % (dim_red, pi, Cfn, Cfp))
    model_evaluator.kfold_cross_validation(GauClf.MVG(), DT, LT, k=3, preproc='raw', dimred=dim_red,  app=selected_app)
    print('R: MVG Classifier\nPreprocessing: -\nApplication: (pi=%.2f, Cfn=%.2f, Cfp=%.2f)\nValidation: single split' % (pi, Cfn, Cfp))
    #model_evaluator.singlefold_validation(GauClf.MVG(), DT, LT, preproc='raw', app=selected_app)
    print('-----------------------------------------')
    
    # Test k-fold cross validation (on MVG classifier)
    print('R: MVG Classifier\nPreprocessing: gaussianization\nDim. Reduction: %s\nApplication: (pi=%.2f, Cfn=%.2f, Cfp=%.2f)' % (dim_red, pi, Cfn, Cfp))
    model_evaluator.kfold_cross_validation(GauClf.MVG(), DT, LT, k=5, preproc='gau', dimred=dim_red, app=selected_app)
    print('R: MVG Classifier\nPreprocessing: gaussianization\nApplication: (pi=%.2f, Cfn=%.2f, Cfp=%.2f)\nValidation: single split' % (pi, Cfn, Cfp))
    #model_evaluator.singlefold_validation(GauClf.MVG(), DT, LT, preproc='gau', app=selected_app)
    print('-----------------------------------------')

    # Test k-fold cross validation (on MVG classifier)
    print('R: MVG Classifier\nPreprocessing: znorm\nDim. Reduction: %s\nApplication: (pi=%.2f, Cfn=%.2f, Cfp=%.2f)' % (dim_red, pi, Cfn, Cfp))
    model_evaluator.kfold_cross_validation(GauClf.MVG(), DT, LT, k=5, preproc='znorm', dimred=dim_red, app=selected_app)
    print('R: MVG Classifier\nPreprocessing: znorm\nApplication: (pi=%.2f, Cfn=%.2f, Cfp=%.2f)\nValidation: single split' % (pi, Cfn, Cfp))
    #model_evaluator.singlefold_validation(GauClf.MVG(), DT, LT, preproc='znorm', app=selected_app)
    print('-----------------------------------------')

    """
    # --- Dimensionality Reduction ---
    pca = DimRed.PCA(DT)
    DTR_pca = pca.fitPCA(2)
    pca.scatter_2D_plot(DTR_pca, LT)

    lda = DimRed.LDA(DT, LT)
    DTR_lda = lda.fitLDA(2, True)
    lda.scatter_2D_plot(DTR_lda, LT)

    # plt.show()

    # --- Gaussian Models ---
    model_evaluator = ModelEvaluation.BinaryModelEvaluator()

    # MVG
    mvg_clf = GauClf.MVG()
    labels = mvg_clf.train(DTR, LTR).predict(DTE, labels=True)
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

    # cross validation of MVG
    mvg_clf_kfold = GauClf.MVG()
    model_evaluator.kfold_cross_validation(model=mvg_clf_kfold, D=DT, L=LT, k=10)
    """


