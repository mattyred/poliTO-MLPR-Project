import numpy as np
import PreProcessing
import matplotlib.pyplot as plt
import DimensionalityReduction as DimRed
import GaussianClassifiers as GauClf
import LogisticRegressionClassifier as LogRegClf


class BinaryModelEvaluator:
    """
    @staticmethod
    def confusion_matrix(predicted_labels, actual_labels):
        i = 0
        conf_matrix = np.zeros(shape=(2, 2))
        for predicted_label in predicted_labels:
            actual_label = actual_labels[i]
            if predicted_label == actual_label:
                conf_matrix[predicted_label][predicted_label] += 1
            else:
                conf_matrix[predicted_label][actual_label] += 1
            i += 1
        return conf_matrix
    """
    @staticmethod
    def accuracy(conf_matrix):
        return (conf_matrix[0, 0] + conf_matrix[1, 1]) / (conf_matrix[0, 0] + conf_matrix[1, 1] + conf_matrix[1, 0] + conf_matrix[0, 1])

    @staticmethod
    def error_rate(predicted_labels, actual_labels):
        return 1 - np.sum([predicted_labels == actual_labels]) / len(predicted_labels)

    @staticmethod
    def confusion_matrix(predicted_labels, actual_labels):
        """
        :param predicted_labels: label predicted for samples according to scores and threshold
        :param actual_labels: actual labels of samples
        :return:
                  0 | 1
               0 TN | FN
               1 FP | TP
        """
        conf_matrix = np.zeros(shape=(2, 2))
        i = 0
        """
        for pl in predicted_labels:
            if pl == actual_labels[i]:
                conf_matrix[pl][pl] += 1
            else:
                conf_matrix[pl][actual_labels[i]] += 1
            i += 1
        """
        for i in range(2):
            for j in range(2):
                conf_matrix[i, j] = np.sum((predicted_labels == i) & (actual_labels == j))
        return conf_matrix

    @staticmethod
    def optimalBayes_confusion_matrix(scores, actual_labels, pi1, Cfn, Cfp):
        # compute threshold t according to pi1(prior probability), Cfn, Cfp of that application
        t = -np.log((pi1 * Cfn) / ((1 - pi1) * Cfp))
        # make predictions using scores of the classifier R and computed thresholds
        predicted_labels = np.array(scores > t, dtype='int32')
        return BinaryModelEvaluator.confusion_matrix(predicted_labels, actual_labels)

    @staticmethod
    def DCFu(scores, actual_labels, pi1, Cfn, Cfp):
        M = BinaryModelEvaluator.optimalBayes_confusion_matrix(scores, actual_labels, pi1, Cfn, Cfp)
        fnr = M[0, 1] / (M[0, 1] + M[1, 1])  # FNR
        fpr = M[1, 0] / (M[0, 0] + M[1, 0])  # FPR
        return pi1 * Cfn * fnr + (1 - pi1) * Cfp * fpr

    @staticmethod
    def DCF(scores, actual_labels, pi1, Cfn, Cfp):
        # Compute the DCF of the best dummy system: R that classifies everything as 1 or everything as 0
        Bdummy_DCF = np.minimum(pi1 * Cfn, (1 - pi1) * Cfp)
        return BinaryModelEvaluator.DCFu(scores, actual_labels, pi1, Cfn, Cfp) / Bdummy_DCF

    @staticmethod
    def minDCF(scores, actual_labels, pi1, Cfn, Cfp):
        # Score in increasing order all the scores produces by classifier R
        scores_sort = np.sort(scores)
        normDCFs = []
        for t in scores_sort:
            # Make prediction by using a threshold t varying among all different sorted scores (all possible thresholds)
            predicted_labels = np.where(scores > t + 0.000001, 1, 0)
            # Compute confusion matrix given those predicted labels
            M = BinaryModelEvaluator.confusion_matrix(predicted_labels, actual_labels)
            # Compute FNR, FPR of that confusion matrix
            fnr = M[0, 1] / (M[0, 1] + M[1, 1])
            fpr = M[1, 0] / (M[0, 0] + M[1, 0])
            # Compute the DCF(normalized) associated to threshold 't' for application (pi1, Cfn, Cfp)
            dcf = pi1 * Cfn * fnr + (1 - pi1) * Cfp * fpr
            Bdummy_DCF = np.minimum(pi1 * Cfn, (1 - pi1) * Cfp)
            dcf_norm = dcf / Bdummy_DCF
            normDCFs.append(dcf_norm)
        return min(normDCFs)

    @staticmethod
    def plotROC(llrs, actual_labels):
        TPR = []
        FPR = []
        llrs_sort = np.sort(llrs)
        for i in llrs_sort:
            predicted_labels = np.where(llrs > i + 0.000001, 1, 0)
            conf_matrix = BinaryModelEvaluator.confusion_matrix(predicted_labels, actual_labels)
            TPR.append(conf_matrix[1, 1] / (conf_matrix[0, 1] + conf_matrix[1, 1]))
            FPR.append(conf_matrix[1, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0]))
        plt.plot(np.array(FPR), np.array(TPR))
        plt.show()

    @staticmethod
    def plot_Bayes_error(scores, actual_labels):
        effPriorLogOdds = np.linspace(-3, 3, 21)
        effPrior = 1 / (1 + np.exp(-effPriorLogOdds))
        mindcf = []
        dcf = []
        for prior in effPrior:
            mindcf.append(BinaryModelEvaluator.minDCF(scores, actual_labels, prior, 1, 1))
            dcf.append(BinaryModelEvaluator.DCF(scores, actual_labels, prior, 1, 1))
        plt.plot(effPriorLogOdds, dcf, label='DCF', color='r')
        plt.plot(effPriorLogOdds, mindcf, label='min DCF', color='b')
        plt.ylim([0, 1.1])
        plt.xlim([-3, 3])
        plt.legend(['DCF', 'min DCF'])
        plt.show()

    @staticmethod
    def kfold_cross_validation(model=None, D=None, L=None, k=10, preproc='raw', dimred=None, iprint=True):
        PreProcesser = PreProcessing.DataPreProcesser()
        Nsamples = D.shape[1]
        np.random.seed(0)
        idx = np.random.permutation(Nsamples)
        folds = np.array_split(idx, k)
        k_scores = []
        k_labels = []
        if dimred is not None:
            dimred_type = dimred['type']
            dimred_m = dimred['m']
        else:
            dimred_type = None

        for i in range(k):
            # Obtain k-1 folds (Dtrain) and 1 validation fold (Dtest)
            fold_test = folds[i]
            Dtest = D[:, fold_test]
            Ltest = L[fold_test]
            folds_train = []
            for j in range(k):
                if j != i:
                    folds_train.append(folds[j])
            Dtrain = D[:, np.array(folds_train).flat]
            Ltrain = L[np.array(folds_train).flat]

            # --- Pre-Processing ---  #
            if preproc == 'gau':
                # Gaussianize features of Dtrain
                Dtrain_normalized = PreProcesser.gaussianized_features_training(Dtrain)
                # Gaussianize features of Dtest
                Dtest_normalized = PreProcesser.gaussianized_features_evaluation(Dtest, Dtrain)
            elif preproc == 'znorm':
                # Z-Normalize features of Dtrain
                Dtrain_normalized = PreProcesser.znormalized_features_training(Dtrain)
                # Z-Normalize features of Dtest
                Dtest_normalized = PreProcesser.znormalized_features_evaluation(Dtest, Dtrain)
            else:
                Dtrain_normalized = Dtrain
                Dtest_normalized = Dtest

            # --- Dimensionality Reduction --- #
            if dimred_type == 'pca':
                pca = DimRed.PCA(Dtrain_normalized)
                Dtrain_normalized_reduced = pca.fitPCA_train(dimred_m)
                Dtest_normalized_reduced = pca.fitPCA_test(Dtest_normalized)
            elif dimred_type == 'lda':
                # TODO
                Dtrain_normalized_reduced = Dtrain_normalized
                Dtest_normalized_reduced = Dtest_normalized
            else:
                Dtrain_normalized_reduced = Dtrain_normalized
                Dtest_normalized_reduced = Dtest_normalized

            # --- Model Training --- #
            k_scores.append(model.train(Dtrain_normalized_reduced, Ltrain).predict(Dtest_normalized_reduced, labels=False))
            k_labels.append(Ltest)

        # --- Model Evaluation (for different applications)--- #
        k_scores = np.hstack(k_scores)
        k_labels = np.hstack(k_labels)
        dcf = []
        min_dcf = []
        priors = [0.1, 0.5, 0.9]
        for prior in priors:
            dcf.append(BinaryModelEvaluator.DCF(k_scores, k_labels, prior, 1, 1))
            min_dcf.append(BinaryModelEvaluator.minDCF(k_scores, k_labels, prior, 1, 1))
        if iprint:
            for i in range(len(priors)): # for each application (prior)
                print('pi=[%.1f] DCF = %.3f - minDCF = %.3f' % (priors[i], dcf[i], min_dcf[i]))
        return dcf, min_dcf

    @staticmethod
    def singlefold_validation(model=None, D=None, L=None, preproc='raw', app=None):
        PreProcesser = PreProcessing.DataPreProcesser()
        nTrain = int(D.shape[1] * 2.0 / 3.0)  # 2/3 of the dataset D are used for training, 1/3 for validation
        np.random.seed(0)
        idx = np.random.permutation(D.shape[1])
        idxTrain = idx[0:nTrain]
        idxTest = idx[nTrain:]
        DTR = D[:, idxTrain]
        DTE = D[:, idxTest]
        LTR = L[idxTrain]
        LTE = L[idxTest]
        pi = app['pi']
        Cfn = app['Cfn']
        Cfp = app['Cfp']
        scores = []
        if preproc == 'gau':
            Dtrain_gaussianized = PreProcesser.gaussianized_features_training(DTR)
            Dtest_gaussianized = PreProcesser.gaussianized_features_evaluation(DTE, DTR)
            scores = model.train(Dtrain_gaussianized, LTR).predict(Dtest_gaussianized, labels=False)
        elif preproc == 'znorm':
            Dtrain_znorm = PreProcesser.znormalized_features_training(DTR)
            Dtest_znorm = PreProcesser.znormalized_features_evaluation(DTE, DTR)
            scores = model.train(Dtrain_znorm, LTR).predict(Dtest_znorm, labels=False)
        elif preproc == 'raw':
            scores = model.train(DTR, LTR).predict(DTE, labels=False)
        dcf_app1 = BinaryModelEvaluator.DCF(scores, LTE, pi, Cfn, Cfp)
        mindcf_app1 = BinaryModelEvaluator.minDCF(scores, LTE, pi, Cfn, Cfp)
        print('DCF = %.3f - minDCF = %.3f\n\n' % (dcf_app1, mindcf_app1))

    @staticmethod
    def validate_final_model(model=None):
        pass

    @staticmethod
    def plot_lambda_minDCF_LinearLogisticRegression(model, DT, LT, k, selected_app, dim_red):
        l = [1.E-6, 1.E-5, 1.E-4, 1.E-3, 1.E-2, 1.E-1, 1, 10, 100, 1000, 10000, 100000]
        plt.figure(figsize=(10, 8))
        plt.xscale('log')
        plt.xticks(l)
        plt.xlabel("λ")
        plt.ylabel("minDCF")
        plt.ylim([0, 1])
        colors = ["red", "blue", "green"]
        minDCF_values_01 = []
        minDCF_values_05 = []
        minDCF_values_09 = []
        for lbd in l:
            model = LogRegClf.LinearLogisticRegression(lbd, prior_weighted=True, prior=0.5)
            DCF, minDCF = BinaryModelEvaluator.kfold_cross_validation(
                    model,
                    DT,
                    LT,
                    k=k,
                    preproc='raw',
                    dimred=dim_red,
                    app=selected_app,
                    iprint=False)
            minDCF_values_01.append(minDCF[0])
            minDCF_values_05.append(minDCF[1])
            minDCF_values_09.append(minDCF[2])

        plt.plot(l, minDCF_values_01, label="π = 0.1", color=colors[0])
        plt.plot(l, minDCF_values_05, label="π = 0.5", color=colors[1])
        plt.plot(l, minDCF_values_09, label="π = 0.9", color=colors[2])
        plt.show()