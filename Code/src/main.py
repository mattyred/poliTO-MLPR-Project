import numpy as np
import matplotlib.pyplot as plt
import DataImport
import DimensionalityReduction as DimRed
import GaussianClassifiers as GauClf
import ModelEvaluation
import PreProcessing
import LogisticRegressionClassifier as LogRegClf
import GMMClassifier as GMMClf

import SVMClassifier as SVMClf
import GMMClassifier
if __name__ == '__main__':
    DT, LT = DataImport.read_data_training('./Dataset/Train.txt')
    DE, LE = DataImport.read_data_evaluation('./Dataset/Test.txt')

    #########################
    ### FEATURES ANALYSIS ###
    #########################
    # Features analysis - overall statistics
    m = np.mean(DT, axis=1).reshape(-1, 1)
    std = np.std(DT, axis=1).reshape(-1, 1)

    # Features analysis - histograms
    PreProcesser = PreProcessing.DataPreProcesser()
    PreProcesser.plot_features_hist(DT, LT, preproc='gau', title=False)

    # Features analysis - correlation of non-preprocessed features
    PreProcesser = PreProcessing.DataPreProcesser()
    DTz = PreProcesser.znormalized_features_training(DT)
    DTzgau = PreProcesser.gaussianized_features_training(DTz)
    PreProcesser.heatmap(DTz, LT, plt, 'Features correlation (no preprocessing)')
    # Features analyssis - correlation of gaussianized features
    PreProcesser.heatmap(DTzgau, LT, plt, 'Features correlation (z-norm + gaussianization)')
    # Features analyssis - correlation of gaussianized features
    PreProcesser.heatmap(DTz, LT, plt, 'Features correlation (z-normalized features)')
    plt.show()
    
    # PCA K-FOLD
    m, t = DimRed.PCA().kfold_PCA(D=DT, k=3, threshold=0.95, show=True)
    
    ############################
    ### GENERATIVE GAUSSIANS ###
    ############################
    # MVG FULL
    model_evaluator = ModelEvaluation.BinaryModelEvaluator()
    dim_red = {'type': 'pca', 'm': 11}
    model_evaluator.plot_gaussian_models(DT=DT, LT=LT)
    print('R: MVG Classifier\nPreprocessing: znorm\nDim. Reduction: %s\nValidation: k-fold' % dim_red)
    model_evaluator.kfold_cross_validation(GauClf.MVG(), DT, LT, k=3, preproc='znorm', dimred=dim_red,  iprint=True)
    
    print('R: MVG Classifier\nPreprocessing: znorm+gau\nDim. Reduction: %s\n' % dim_red)
    model_evaluator.kfold_cross_validation(GauClf.MVG(), DT, LT, k=3, preproc='zg', dimred=dim_red)

    # MVG TIED

    print('R: MVG-Tied Classifier\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    model_evaluator.kfold_cross_validation(GauClf.TiedG(), DT, LT, k=3, preproc='znorm', dimred=dim_red)

    print('R: MVG-Tied Classifier\nPreprocessing: znorm+gau\nDim. Reduction: %s\n' % dim_red)
    model_evaluator.kfold_cross_validation(GauClf.TiedG(), DT, LT, k=5, preproc='zg', dimred=dim_red)

    # NAIVE BAYES
    print('R: Naive Bayes Classifier\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    model_evaluator.kfold_cross_validation(GauClf.NaiveBayes(), DT, LT, k=3, preproc='znorm', dimred=dim_red)

    print('R: Naive Bayes Classifier\nPreprocessing: znorm+gau\nDim. Reduction: %s\n' % dim_red)
    model_evaluator.kfold_cross_validation(GauClf.NaiveBayes(), DT, LT, k=3, preproc='zg', dimred=dim_red)

    ##################################
    ### LINEAR LOGISTIC REGRESSION ###
    ##################################
    
    model_evaluator = ModelEvaluation.BinaryModelEvaluator()
    dim_red = None#{'type': 'pca', 'm': 11}

    lbd = 10**-3
    print('R: Linear Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    model_evaluator.kfold_cross_validation(LogRegClf.LinearLogisticRegression(lbd, prior_weighted=True, prior=0.5),
                                           DT,
                                           LT,
                                           k=3,
                                           preproc='znorm',
                                           dimred=dim_red)
    #ModelEvaluation.BinaryModelEvaluator().plot_lambda_minDCF_LinearLogisticRegression(DT=DT, LT=LT, k=3)

    #####################################
    ### QUADRATIC LOGISTIC REGRESSION ###
    #####################################

    model_evaluator = ModelEvaluation.BinaryModelEvaluator()
    dim_red = {'type': 'pca', 'm': 10}
    lbd = 10**-6
    print('R: Quadric Logistic Regression\nPreprocessing: znorm\nDim. Reduction: %s\n' % dim_red)
    model_evaluator.kfold_cross_validation(LogRegClf.QuadraticLogisticRegression(lbd, prior_weighted=False),
                                           DT,
                                           LT,
                                           k=3,
                                           preproc='znorm',
                                           dimred=dim_red)

    #ModelEvaluation.BinaryModelEvaluator().plot_lambda_minDCF_QuadraticLogisticRegression(DT=DT, LT=LT, k=3)
   
    ##################
    ### LINEAR SVM ###
    ##################

    model_evaluator = ModelEvaluation.BinaryModelEvaluator()
    hparams = {'K': 10, 'eps': 1, 'gamma': 1, 'C': 10}
    dim_red = None#{'type': 'pca', 'm': 9}
    model_evaluator.plot_lambda_minDCF_LinearSVM(DT, LT, 3)
    print('R: SVM Linear\nPreprocessing: znorm\nDim. Reduction: %s\nHyperparameters: %s' % (dim_red, hparams))
    model_evaluator.kfold_cross_validation(SVMClf.SVM(hparams, None),
                                           DT,
                                           LT,
                                           k=3,
                                           preproc='znorm',
                                           dimred=dim_red,
                                           iprint=True)
    #############################
    ### PLOYNOMIAL KERNEL SVM ###
    #############################

    model_evaluator = ModelEvaluation.BinaryModelEvaluator()
    hparams = {'K': 0, 'eps': 0, 'gamma': 1, 'C': 1, 'c': 0, 'd': 1}
    # hparams = {'K': 0, 'eps': 0, 'gamma': 10**-3, 'C': 10**-1, 'c': 0, 'd': 1} 
    # hparams = {'K': 0, 'eps': 0, 'gamma': 10 ** -3, 'C': 10 ** -1, 'c': 0, 'd': 1} 
    dim_red = None#{'type': 'pca', 'm': 8}
    model_evaluator.kfold_cross_validation(SVMClf.SVM(hparams, kernel='Polynomial', prior=0.5),
                                           DT,
                                           LT,
                                           k=3,
                                           preproc='raw',
                                           dimred=dim_red,
                                           iprint=True)
    ######################
    ### RBF KERNEL SVM ###
    ######################

    model_evaluator = ModelEvaluation.BinaryModelEvaluator()
    #hparams = {'K': 0, 'eps': 0, 'gamma': 1, 'C': 1, 'c': 0, 'd': 1}
    #hparams = {'K': 0, 'eps': 0, 'gamma': 10**-3, 'C': 10**-1, 'c': 0, 'd': 1} 
    hparams = {'K': 0, 'eps': 0, 'gamma': 10 ** -3, 'C': 10 ** -1, 'c': 0, 'd': 1}
    dim_red = None#{'type': 'pca', 'm': 8}
    model_evaluator.kfold_cross_validation(SVMClf.SVM(hparams, kernel='RBF', prior=0.5),
                                           DT,
                                           LT,
                                           k=3,
                                           preproc='raw',
                                           dimred=dim_red,
                                           iprint=True)

    # SVM PLOTS
    model_evaluator = ModelEvaluation.BinaryModelEvaluator()
    dim_red = None  # {'type': 'pca', 'm': 9}
    model_evaluator.plot_lambda_minDCF_LinearSVM(DT, LT, 3, dim_red)

    ###########
    ### GMM ###
    ###########

    model_evaluator = ModelEvaluation.BinaryModelEvaluator()
    dim_red = {'type': 'pca', 'm': 10}
    nComponents = 8
    cov = 'Diag'
    print('R: GMM Classifier(%d components - %s cov)\nPreprocessing: znorm\nDim. Reduction: %s\n' % (nComponents, cov, dim_red))
    model_evaluator.kfold_cross_validation(GMMClassifier.GMM(alpha=0.1, nComponents=nComponents, psi=0.01, covType=cov),
                                           DT,
                                           LT,
                                           k=3,
                                           preproc='zg',
                                           dimred=dim_red)

    #ModelEvaluation.BinaryModelEvaluator().plot_histogramGMM(DT, LT)
    
    #########################
    ### SCORE CALIBRATION ###
    ########################

    fig1, axs1 = plt.subplots(2, 2)
    fig1.set_figheight(5)
    fig1.set_figwidth(13)
    model_evaluator.plot_Bayes_error(ax=axs1[0, 0], title='Gaussian Tied Covariance',
                                        model=GauClf.TiedG(), preproc='znorm',
                                        dimred=None, DT=DT, LT=LT, calibrate_scores=False)
    model_evaluator.plot_Bayes_error(ax=axs1[0, 1], title='Logistic Regression',
                                        model=LogRegClf.LinearLogisticRegression(lbd=10**-6), preproc='znorm',
                                        dimred=None, DT=DT, LT=LT, calibrate_scores=False)
    model_evaluator.plot_Bayes_error(ax=axs1[1, 0], title='SVM',
                                        model=SVMClf.SVM(hparams={'K': 0, 'C': 1}), preproc='znorm',
                                        dimred=None, DT=DT, LT=LT, calibrate_scores=False)
    model_evaluator.plot_Bayes_error(ax=axs1[1, 1], title='GMM',
                                        model=GMMClf.GMM(alpha=0.1, nComponents=8, psi=0.01, covType='Tied'), preproc='znorm',
                                        dimred=None, DT=DT, LT=LT, calibrate_scores=False)
    fig1.tight_layout()
    plt.show()

    fig1, axs1 = plt.subplots(2, 2)
    fig1.set_figheight(5)
    fig1.set_figwidth(13)
    model_evaluator.plot_Bayes_error(ax=axs1[0, 0], title='Gaussian Tied Covariance',
                                        model=GauClf.TiedG(), preproc='znorm',
                                        dimred=None, DT=DT, LT=LT, calibrate_scores=True)
    model_evaluator.plot_Bayes_error(ax=axs1[0, 1], title='Logistic Regression',
                                        model=LogRegClf.LinearLogisticRegression(lbd=10**-6), preproc='znorm',
                                        dimred=None, DT=DT, LT=LT, calibrate_scores=True)
    model_evaluator.plot_Bayes_error(ax=axs1[1, 0], title='SVM',
                                        model=SVMClf.SVM(hparams={'K': 0, 'C': 1}), preproc='znorm',
                                        dimred=None, DT=DT, LT=LT, calibrate_scores=True)
    model_evaluator.plot_Bayes_error(ax=axs1[1, 1], title='GMM',
                                        model=GMMClf.GMM(alpha=0.1, nComponents=8, psi=0.01, covType='Tied'), preproc='znorm',
                                        dimred=None, DT=DT, LT=LT, calibrate_scores=True)
    fig1.tight_layout()
    plt.show()

    #################
    ### ROC CURVE ###
    #################
    hparams = {'K': 0, 'eps': 0, 'gamma': 10 ** -3, 'C': 10 ** -1, 'c': 0, 'd': 1}
    model_evaluator = ModelEvaluation.BinaryModelEvaluator()
    fig = plt.figure()
    model_evaluator.plotROCs(plt=plt, model=GauClf.TiedG(), preproc='znorm', DT=DT, LT=LT, DE=DE, LE=LE)
    model_evaluator.plotROCs(plt=plt, model=LogRegClf.LinearLogisticRegression(lbd=10**-6), preproc='znorm', DT=DT, LT=LT, DE=DE, LE=LE)
    model_evaluator.plotROCs(plt=plt, model=SVMClf.SVM(hparams={'K': 0, 'C': 1}), preproc='znorm', DT=DT, LT=LT, DE=DE, LE=LE)
    model_evaluator.plotROCs(plt=plt, model=GMMClassifier.GMM(alpha=0.1, nComponents=8, psi=0.01, covType='Tied'), preproc='znorm', DT=DT, LT=LT, DE=DE,
                             LE=LE)
    fig.legend(['Tied Gaussian', 'Linear LR', 'Linear SVM', 'Tied GMM(9 components)'])
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()