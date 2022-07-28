import numpy as np
import matplotlib.pyplot as plt
import DataImport
import Constants as CONST
import DimensionalityReduction as DimRed
import GaussianClassifiers as GauClf
import ModelEvaluation
import PreProcessing
import LogisticRegressionClassifier as LogRegClf
import LDAClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import scipy.optimize
from sklearn import svm
from sklearn.mixture import GaussianMixture

import SVMClassifier as SVMClf
import GMMClassifier
if __name__ == '__main__':
    DT, LT = DataImport.read_data_training('./Dataset/Train.txt')
    DE, LE = DataImport.read_data_evaluation('./Dataset/Test.txt')

    """
    # Features analysis - overall statistics
    m = np.mean(DT, axis=1).reshape(-1, 1)
    std = np.std(DT, axis=1).reshape(-1, 1)
    """
    """
    # Features analysis - histograms
    PreProcesser = PreProcessing.DataPreProcesser()
    PreProcesser.plot_features_hist(DT, LT, preproc='gau', title=False)
    """
    """
    # Features analysis - correlation of non-preprocessed features
    PreProcesser = PreProcessing.DataPreProcesser()
    DTz = PreProcesser.znormalized_features_training(DT)
    DTzgau = PreProcesser.gaussianized_features_training(DTz)
    PreProcesser.heatmap(DTzgau, LT, plt, 'Features correlation (no preprocessing)')
    # Features analyssis - correlation of gaussianized features
    #PreProcesser.heatmap(DTzgau, LT, plt, 'Features correlation (z-norm + gaussianization)')
    # Features analyssis - correlation of gaussianized features
    #PreProcesser.heatmap(DTz, LT, plt, 'Features correlation (z-normalized features)')
    plt.show()
    """

    """
    # PCA K-FOLD
    m, t = DimRed.PCA().kfold_PCA(D=DT, k=3, threshold=0.95, show=True)
    """

    """
    # Test k-fold cross validation (on MVG classifier)
    model_evaluator = ModelEvaluation.BinaryModelEvaluator()
    dim_red = None#{'type': 'pca', 'm': 9}
    model_evaluator.plot_gaussian_models(DT=DT, LT=LT)
    print('R: MVG Classifier\nPreprocessing: znorm\nDim. Reduction: %s\nValidation: k-fold' % dim_red)
    model_evaluator.kfold_cross_validation(GauClf.MVG(), DT, LT, k=3, preproc='znorm', dimred=dim_red,  iprint=True)
    print('-----------------------------------------')
    
    # Test k-fold cross validation (on MVG classifier)
    print('R: MVG Classifier\nPreprocessing: znorm+gau\nDim. Reduction: %s\n' % dim_red)
    model_evaluator.kfold_cross_validation(GauClf.MVG(), DT, LT, k=3, preproc='zg', dimred=dim_red)
    print('-----------------------------------------')

    # Test k-fold cross validation (on MVG Tied Classifier)

    print('R: MVG-Tied Classifier\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    model_evaluator.kfold_cross_validation(GauClf.TiedG(), DT, LT, k=3, preproc='znorm', dimred=dim_red)
    print('-----------------------------------------')

    print('R: MVG-Tied Classifier\nPreprocessing: znorm+gau\nDim. Reduction: %s\n' % dim_red)
    model_evaluator.kfold_cross_validation(GauClf.TiedG(), DT, LT, k=5, preproc='zg', dimred=dim_red)
    print('-----------------------------------------')

    print('-----------------------------------------')
    # Test k-fold cross validation (on MVG Naive Bayes Classifier)
    print('R: Naive Bayes Classifier\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    model_evaluator.kfold_cross_validation(GauClf.NaiveBayes(), DT, LT, k=3, preproc='znorm', dimred=dim_red)
    print('-----------------------------------------')

    print('R: Naive Bayes Classifier\nPreprocessing: znorm+gau\nDim. Reduction: %s\n' % dim_red)
    model_evaluator.kfold_cross_validation(GauClf.NaiveBayes(), DT, LT, k=3, preproc='zg', dimred=dim_red)
    print('-----------------------------------------')
    """
    """
    pca = DimRed.PCA(DT)
    DTpca = pca.fitPCA_train(2)
    DTpca_m = DTpca[:, LT == 0]
    DTpca_f = DTpca[:, LT == 1]
    plt.figure()
    plt.scatter(DTpca_m[0, :], DTpca_m[1, :], alpha=.3, marker='o', edgecolors='blue', color='none')  # male
    plt.scatter(DTpca_f[0, :], DTpca_f[1, :], alpha=.3, marker='o', edgecolors='red', color='none')   # female
    plt.show()
    """

    # LOGISTIC REGRESSION
    """
    model_evaluator = ModelEvaluation.BinaryModelEvaluator()
    dim_red = {'type': 'pca', 'm': 10}

    lbd = 10**-3
    print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    model_evaluator.kfold_cross_validation(LogRegClf.LinearLogisticRegression(lbd, prior_weighted=False, prior=0.5),
                                           DT,
                                           LT,
                                           k=3,
                                           preproc='znorm',
                                           dimred=dim_red)
    """
    #model_evaluator.plot_lambda_minDCF_LinearLogisticRegression(DT=DT, LT=LT, k=3)
    """
    # QUADRATIC LOGISTIC REGRESSION
    model_evaluator = ModelEvaluation.BinaryModelEvaluator()
    dim_red = {'type': 'pca', 'm': 9}
    lbd = 10**-6
    print('R: Quadric Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    model_evaluator.kfold_cross_validation(LogRegClf.QuadraticLogisticRegression(lbd, prior_weighted=False),
                                           DT,
                                           LT,
                                           k=3,
                                           preproc='znorm',
                                           dimred=dim_red)
    """
    #ModelEvaluation.BinaryModelEvaluator().plot_lambda_minDCF_QuadraticLogisticRegression(DT=DT, LT=LT, k=3)

    # SVM LINEAR
    model_evaluator = ModelEvaluation.BinaryModelEvaluator()
    hparams = {'K': 10, 'eps': 1, 'gamma': 1, 'C': 10}
    dim_red = None#{'type': 'pca', 'm': 9}
    model_evaluator.plot_lambda_minDCF_LinearSVM(DT, LT, 3)
    """
    print('R: SVM Linear\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    model_evaluator.kfold_cross_validation(SVMClf.SVM(hparams, None),
                                           DT,
                                           LT,
                                           k=3,
                                           preproc='znorm',
                                           dimred=dim_red,
                                           iprint=True)
    """
    # SVM KERNEL POLY

    # SVM KERNEL RBF
    """
    model_evaluator = ModelEvaluation.BinaryModelEvaluator()
    #hparams = {'K': 0, 'eps': 0, 'gamma': 1, 'C': 1, 'c': 0, 'd': 1}
    #hparams = {'K': 0, 'eps': 0, 'gamma': 10**-3, 'C': 10**-1, 'c': 0, 'd': 1} 0.158 / 0.059 / 0.144
    hparams = {'K': 0, 'eps': 0, 'gamma': 10 ** -3, 'C': 10 ** -1, 'c': 0, 'd': 1} # 0.168 / 0.061 / 0.148
    dim_red = None#{'type': 'pca', 'm': 8}
    model_evaluator.kfold_cross_validation(SVMClf.SVM(hparams, kernel='RBF', prior=0.5),
                                           DT,
                                           LT,
                                           k=3,
                                           preproc='raw',
                                           dimred=dim_red,
                                           iprint=True)


    """
    #clf = svm.SVC(kernel='poly', degree=2, coef0=1, C=1)
    #clf.fit(DT.T, LT)
    #print(sum(clf.predict(DE.T) == LE) / len(LE))
    """
    # SVM PLOTS
    """
    #model_evaluator = ModelEvaluation.BinaryModelEvaluator()
    #dim_red = None  # {'type': 'pca', 'm': 9}
    #model_evaluator.plot_lambda_minDCF_LinearSVM(DT, LT, 3, dim_red)
    """
    
    """

    """
    # LDA CLASSIFIER VS MVG TIED CLASSIFIER
    model_evaluator = ModelEvaluation.BinaryModelEvaluator()
    dim_red = None#{'type': 'pca', 'm': 4}

    print('R: LDA Classifier\nPreprocessing: -\nDim. Reduction: %s\n'%dim_red)
    model_evaluator.kfold_cross_validation(LDAClassifier.LDA(),
                                           DT,
                                           LT,
                                           k=3,
                                           preproc='zg',
                                           dimred=dim_red)
    """
    """
    model = GauClf.TiedG()
    scores_tied = model.train(DT, LT).predict(DE, labels=False)
    M_tied = model_evaluator.optimalBayes_confusion_matrix(scores_tied,
                                                      LE,
                                                      selected_app['pi'],
                                                      selected_app['Cfn'],
                                                      selected_app['Cfp'])
    print('Accuracy: ', model_evaluator.accuracy(M_tied))

    b_tied, c = model.get_decision_function_parameters()
    print('MVG Tied decision function parameters: b=', b_tied,' c=', c)

    model = LDAClassifier.LDA()
    scores_lda = model.train(DT, LT).predict(DE, labels=False)
    M_lda = model_evaluator.optimalBayes_confusion_matrix(scores_lda,
                                                      LE,
                                                      selected_app['pi'],
                                                      selected_app['Cfn'],
                                                      selected_app['Cfp'])
    fat = np.abs(scores_lda / scores_tied)
    print('Accuracy: ', model_evaluator.accuracy(M_lda))
    b_lda = model.get_decision_function_parameters()
    print('LDA decision function parameters: b=', b_lda)
    """

    """
    # PCA K-FOLD
    m, t = DimRed.PCA().kfold_PCA(D=DT, k=3, threshold=0.95, show=True)
    """

    # GMM
    """
    model_evaluator = ModelEvaluation.BinaryModelEvaluator()
    dim_red = None#{'type': 'pca', 'm': 9}
    nComponents = 8
    cov = 'Full'
    print('R: GMM Classifier(%d components - %s cov)\nPreprocessing: znorm\nDim. Reduction: %s\n' % (nComponents, cov, dim_red))
    model_evaluator.kfold_cross_validation(GMMClassifier.GMM(alpha=0.1, nComponents=nComponents, psi=0.01, covType=cov),
                                           DT,
                                           LT,
                                           k=3,
                                           preproc='zg',
                                           dimred=dim_red)
    """
