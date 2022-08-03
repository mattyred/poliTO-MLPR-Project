import numpy as np
import PreProcessing
import matplotlib.pyplot as plt
import DimensionalityReduction as DimRed
import GaussianClassifiers as GauClf
import LogisticRegressionClassifier as LogRegClf
import SVMClassifier as SVMClf
import GMMClassifier


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
    def plot_Bayes_error(ax=None, title=None, model=None, preproc='raw', dimred=None, DT=None, LT=None, calibrate_scores=False):
        effPriorLogOdds = np.linspace(-3, 3, 21)
        effPrior = 1 / (1 + np.exp(-effPriorLogOdds))
        mindcf = []
        dcf = []
        for prior in effPrior:
            DCF, minDCF = BinaryModelEvaluator().kfold_cross_validation(model, DT, LT, k=3, preproc=preproc, dimred=dimred, prior=prior, calibrated=calibrate_scores)
            mindcf.append(minDCF)
            dcf.append(DCF)
        ax.plot(effPriorLogOdds, dcf, label='DCF', color='red')
        ax.plot(effPriorLogOdds, mindcf, label='min DCF', color='blue', linestyle='dashed')
        #plt.ylim([0, 1.1])
        ax.set_xlim([-3, 3])
        ax.legend(['DCF', 'min DCF'])
        ax.set_xlabel("prior log-odds")
        ax.set_title(title)

    @staticmethod
    def kfold_cross_validation(model=None, D=None, L=None, k=10, preproc='raw', dimred=None, iprint=True, prior=None, calibrated=False):
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
            elif preproc == 'zg':
                Dtrain_z = PreProcesser.znormalized_features_training(Dtrain)
                Dtrain_normalized = PreProcesser.gaussianized_features_training(Dtrain_z)
                Dtest_z = PreProcesser.znormalized_features_evaluation(Dtest, Dtrain)
                Dtest_normalized = PreProcesser.gaussianized_features_evaluation(Dtest_z, Dtrain_normalized)
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

        # --- Score Calibration--- #
        if calibrated is True:
            # Train a logistic regression model prior weighted with k_scores as samples
            s = k_scores.reshape(1, -1)
            p = 0.5
            lr = LogRegClf.LinearLogisticRegression(lbd=10**-6, prior_weighted=True, prior=p)
            alpha, betafirst = lr.train(s, k_labels).get_model_parameters()
            k_scores_cal = (alpha*s + betafirst - np.log(p/(1-p))).reshape(Nsamples,)
        else:
            k_scores_cal = k_scores

        dcf = []
        min_dcf = []
        priors = [0.1, 0.5, 0.9]
        if prior is None:
            for p in priors:
                dcf.append(BinaryModelEvaluator.DCF(k_scores_cal, k_labels, p, 1, 1))
                min_dcf.append(BinaryModelEvaluator.minDCF(k_scores_cal, k_labels, p, 1, 1))
        else:  # For Bayes error plot
            return BinaryModelEvaluator.DCF(k_scores_cal, k_labels, prior, 1, 1), BinaryModelEvaluator.minDCF(k_scores_cal, k_labels, prior, 1, 1)

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
    def plot_lambda_minDCF_LinearLogisticRegression(DT=None, LT=None, k=3):
        lambdas = [1.E-6, 1.E-5, 1.E-4, 1.E-3, 1.E-2, 1.E-1, 1, 10, 100, 1000, 10000, 100000]
        fig1, axs1 = plt.subplots(1, 3)
        fig1.set_figheight(5)
        fig1.set_figwidth(13)
        colors = ["red", "blue", "green"]
        i = 0
        for dim_red in [None, {'type': 'pca', 'm': 11}, {'type': 'pca', 'm': 10}]:
            minDCF_values_01 = []
            minDCF_values_05 = []
            minDCF_values_09 = []
            for lbd in lambdas:
                model = LogRegClf.LinearLogisticRegression(lbd, prior_weighted=False, prior=0.5)
                DCF, minDCF = BinaryModelEvaluator.kfold_cross_validation(
                        model,
                        DT,
                        LT,
                        k=k,
                        preproc='znorm',
                        dimred=dim_red,
                        iprint=False)
                minDCF_values_01.append(minDCF[0])
                minDCF_values_05.append(minDCF[1])
                minDCF_values_09.append(minDCF[2])

            axs1[i].plot(lambdas, minDCF_values_01, label="π = 0.1", color=colors[0])
            axs1[i].plot(lambdas, minDCF_values_05, label="π = 0.5", color=colors[1])
            axs1[i].plot(lambdas, minDCF_values_09, label="π = 0.9", color=colors[2])
            if dim_red is None:
                axs1[i].set_title('no PCA')
            else:
                axs1[i].set_title('PCA(m=%d)' % (dim_red['m']))
            axs1[i].set_xscale('log')
            axs1[i].set_xticks([1.E-6, 1.E-4, 1.E-2, 1, 100, 10000, 100000])
            axs1[i].set_xlabel("λ")
            axs1[i].set_ylim([0, 1])
            i += 1
        axs1[0].set_ylabel("minDCF")
        fig1.legend(['π = 0.1', 'π = 0.5', 'π = 0.9'], loc='lower right')
        plt.show()

    @staticmethod
    def plot_lambda_minDCF_QuadraticLogisticRegression(DT=None, LT=None, k=3):
        lambdas = [1.E-6, 1.E-5, 1.E-4, 1.E-3, 1.E-2, 1.E-1, 1, 10, 100, 1000, 10000, 100000]
        fig1, axs1 = plt.subplots(1, 3)
        fig1.set_figheight(5)
        fig1.set_figwidth(13)
        colors = ["red", "blue", "green"]
        i = 0
        for dim_red in [None, {'type': 'pca', 'm': 11}, {'type': 'pca', 'm': 10}]:
            minDCF_values_01 = []
            minDCF_values_05 = []
            minDCF_values_09 = []
            for lbd in lambdas:
                model = LogRegClf.QuadraticLogisticRegression(lbd, prior_weighted=False)
                DCF, minDCF = BinaryModelEvaluator.kfold_cross_validation(
                    model,
                    DT,
                    LT,
                    k=k,
                    preproc='znorm',
                    dimred=dim_red,
                    iprint=False)
                minDCF_values_01.append(minDCF[0])
                minDCF_values_05.append(minDCF[1])
                minDCF_values_09.append(minDCF[2])

            axs1[i].plot(lambdas, minDCF_values_01, label="π = 0.1", color=colors[0])
            axs1[i].plot(lambdas, minDCF_values_05, label="π = 0.5", color=colors[1])
            axs1[i].plot(lambdas, minDCF_values_09, label="π = 0.9", color=colors[2])
            if dim_red is None:
                axs1[i].set_title('no PCA')
            else:
                axs1[i].set_title('PCA(m=%d)' % (dim_red['m']))
            axs1[i].set_xscale('log')
            axs1[i].set_xticks([1.E-6, 1.E-4, 1.E-2, 1, 100, 10000, 100000])
            axs1[i].set_xlabel("λ")
            axs1[i].set_ylim([0, 1])
            i += 1
        axs1[0].set_ylabel("minDCF")
        fig1.legend(['π = 0.1', 'π = 0.5', 'π = 0.9'], loc='lower right')
        plt.show()

    @staticmethod
    def plot_lambda_minDCF_LinearSVM(DT=None, LT=None, k=3):
        C = [1.E-4, 1.E-3, 1.E-2, 1.E-1, 1, 10]
        fig1, axs1 = plt.subplots(1, 3)
        fig1.set_figheight(5)
        fig1.set_figwidth(13)
        colors = ["red", "blue", "green"]
        i = 0
        for dim_red in [None, {'type': 'pca', 'm': 11}, {'type': 'pca', 'm': 10}]:
            minDCF_values_01 = []
            minDCF_values_05 = []
            minDCF_values_09 = []
            for c in C:
                hparams = {'K': 1, 'eps': 1, 'gamma': 1, 'C': c}
                model = SVMClf.SVM(hparams, None)
                DCF, minDCF = BinaryModelEvaluator.kfold_cross_validation(
                    model,
                    DT,
                    LT,
                    k=k,
                    preproc='znorm',
                    dimred=dim_red,
                    iprint=False)
                minDCF_values_01.append(minDCF[0])
                minDCF_values_05.append(minDCF[1])
                minDCF_values_09.append(minDCF[2])

            axs1[i].plot(C, minDCF_values_01, label="π = 0.1", color=colors[0])
            axs1[i].plot(C, minDCF_values_05, label="π = 0.5", color=colors[1])
            axs1[i].plot(C, minDCF_values_09, label="π = 0.9", color=colors[2])
            axs1[i].set_xscale('log')
            axs1[i].set_xticks(C)
            axs1[i].set_xlabel("λ")
            axs1[i].set_ylim([0, 1])
            if dim_red is None:
                axs1[i].set_title('no PCA')
            else:
                axs1[i].set_title('PCA(m=%d)' % (dim_red['m']))
            i += 1
        axs1[0].set_ylabel("minDCF")
        fig1.legend(['π = 0.1', 'π = 0.5', 'π = 0.9'], loc='lower right')
        plt.show()

    @staticmethod
    def plot_lambda_minDCF_RBFSVM(DT=None, LT=None, k=3):
        C = [1.E-4, 1.E-3, 1.E-2, 1.E-1, 1, 10, 10**2]
        fig1, axs1 = plt.subplots(1, 2)
        fig1.set_figheight(5)
        fig1.set_figwidth(13)
        colors = ["red", "blue", "green"]
        i = 0
        for dim_red in [None, {'type': 'pca', 'm': 10}]:
            minDCF_values_01_g01 = []
            minDCF_values_05_g01 = []
            minDCF_values_09_g01 = []
            minDCF_values_01_g001 = []
            minDCF_values_05_g001 = []
            minDCF_values_09_g001 = []
            for c in C:
                # gamma = 0.1
                hparams = {'K': 1, 'eps': 1, 'gamma': 0.1, 'C': c, 'c': 0, 'd': 2}
                model = SVMClf.SVM(hparams, kernel='RBF')
                DCF, minDCF = BinaryModelEvaluator.kfold_cross_validation(
                    model,
                    DT,
                    LT,
                    k=k,
                    preproc='znorm',
                    dimred=dim_red,
                    iprint=False)
                minDCF_values_01_g01.append(minDCF[0])
                minDCF_values_05_g01.append(minDCF[1])
                minDCF_values_09_g01.append(minDCF[2])
                # gamma = 0.01
                hparams = {'K': 1, 'eps': 1, 'gamma': 0.01, 'C': c, 'c': 0, 'd': 2}
                model = SVMClf.SVM(hparams, kernel='RBF')
                DCF, minDCF = BinaryModelEvaluator.kfold_cross_validation(
                    model,
                    DT,
                    LT,
                    k=k,
                    preproc='znorm',
                    dimred=dim_red,
                    iprint=False)
                minDCF_values_01_g001.append(minDCF[0])
                minDCF_values_05_g001.append(minDCF[1])
                minDCF_values_09_g001.append(minDCF[2])

            axs1[i].plot(C, minDCF_values_01_g01, label="π = 0.1", color=colors[0])
            axs1[i].plot(C, minDCF_values_01_g001, label="π = 0.1", color=colors[0], linestlye='dashed')
            axs1[i].plot(C, minDCF_values_05_g01, label="π = 0.5", color=colors[1])
            axs1[i].plot(C, minDCF_values_05_g001, label="π = 0.5", color=colors[1], linestlye='dashed')
            axs1[i].plot(C, minDCF_values_09_g01, label="π = 0.9", color=colors[2])
            axs1[i].plot(C, minDCF_values_09_g001, label="π = 0.9", color=colors[2], linestlye='dashed')
            axs1[i].set_xscale('log')
            axs1[i].set_xticks(C)
            axs1[i].set_xlabel("C")
            axs1[i].set_ylim([0, 1.2])
            if dim_red is None:
                axs1[i].set_title('no PCA')
            else:
                axs1[i].set_title('PCA(m=%d)' % (dim_red['m']))
            i += 1
        axs1[0].set_ylabel("minDCF")
        fig1.legend(['π = 0.1 - γ = 0.1', 'π = 0.1 - γ = 0.01',
                     'π = 0.5 - γ = 0.1', 'π = 0.5 - γ = 0.01',
                     'π = 0.9 - γ = 0.1', 'π = 0.9 - γ = 0.01'], loc='lower right')
        plt.show()

    @staticmethod
    def plot_gaussian_models(DT, LT):
        # Z-Normalized
        fig1, axs1 = plt.subplots(2, 3)
        x = [0.1, 0.5, 0.9]
        _, mindcf_mvg = BinaryModelEvaluator.kfold_cross_validation(GauClf.MVG(), DT, LT, k=3, preproc='znorm', dimred=None, iprint=False)
        _, mindcf_tied = BinaryModelEvaluator.kfold_cross_validation(GauClf.TiedG(), DT, LT, k=3, preproc='znorm', dimred=None, iprint=False)
        _, mindcf_naive = BinaryModelEvaluator.kfold_cross_validation(GauClf.NaiveBayes(), DT, LT, k=3, preproc='znorm', dimred=None, iprint=False)
        axs1[0, 0].plot(x, mindcf_mvg, '--o', label='Full')
        axs1[0, 0].plot(x, mindcf_tied, '--o', label='Tied')
        axs1[0, 0].plot(x, mindcf_naive, '--o', label='Naive')
        axs1[0, 0].set_ylim(0, 1)
        axs1[0, 0].set_title('no PCA')
        axs1[0, 0].set_xticks([0.1, 0.5, 0.9])
        axs1[0, 0].set_xlabel("π")
        dim_red = {'type': 'pca', 'm': 11}
        _, mindcf_mvg = BinaryModelEvaluator.kfold_cross_validation(GauClf.MVG(), DT, LT, k=3, preproc='znorm', dimred=dim_red, iprint=False)
        _, mindcf_tied = BinaryModelEvaluator.kfold_cross_validation(GauClf.TiedG(), DT, LT, k=3, preproc='znorm', dimred=dim_red, iprint=False)
        _, mindcf_naive = BinaryModelEvaluator.kfold_cross_validation(GauClf.NaiveBayes(), DT, LT, k=3, preproc='znorm', dimred=dim_red, iprint=False)
        axs1[0, 1].plot(x, mindcf_mvg, '--o', label='Full')
        axs1[0, 1].plot(x, mindcf_tied, '--o', label='Tied')
        axs1[0, 1].plot(x, mindcf_naive, '--o', label='Naive')
        axs1[0, 1].set_ylim(0, 1)
        axs1[0, 1].set_title('PCA(m=11)')
        axs1[0, 1].set_xticks([0.1, 0.5, 0.9])
        axs1[0, 1].set_xlabel("π")
        dim_red = {'type': 'pca', 'm': 10}
        _, mindcf_mvg = BinaryModelEvaluator.kfold_cross_validation(GauClf.MVG(), DT, LT, k=3, preproc='znorm', dimred=dim_red, iprint=False)
        _, mindcf_tied = BinaryModelEvaluator.kfold_cross_validation(GauClf.TiedG(), DT, LT, k=3, preproc='znorm', dimred=dim_red, iprint=False)
        _, mindcf_naive = BinaryModelEvaluator.kfold_cross_validation(GauClf.NaiveBayes(), DT, LT, k=3, preproc='znorm', dimred=dim_red, iprint=False)
        axs1[0, 2].plot(x, mindcf_mvg, '--o', label='Full')
        axs1[0, 2].plot(x, mindcf_tied, '--o', label='Tied')
        axs1[0, 2].plot(x, mindcf_naive, '--o', label='Naive')
        axs1[0, 2].set_ylim(0, 1)
        axs1[0, 2].set_title('PCA(m=10)')
        axs1[0, 2].set_xticks([0.1, 0.5, 0.9])
        axs1[0, 2].set_xlabel("π")
        # Gaussianized
        _, mindcf_mvg = BinaryModelEvaluator.kfold_cross_validation(GauClf.MVG(), DT, LT, k=3, preproc='zg', dimred=None, iprint=False)
        _, mindcf_tied = BinaryModelEvaluator.kfold_cross_validation(GauClf.TiedG(), DT, LT, k=3, preproc='zg', dimred=None, iprint=False)
        _, mindcf_naive = BinaryModelEvaluator.kfold_cross_validation(GauClf.NaiveBayes(), DT, LT, k=3, preproc='zg', dimred=None, iprint=False)
        axs1[1, 0].plot(x, mindcf_mvg, '--o', label='Full')
        axs1[1, 0].plot(x, mindcf_tied, '--o', label='Tied')
        axs1[1, 0].plot(x, mindcf_naive, '--o', label='Naive')
        axs1[1, 0].set_xticks([0.1, 0.5, 0.9])
        axs1[1, 0].set_xlabel("π")
        dim_red = {'type': 'pca', 'm': 11}
        _, mindcf_mvg = BinaryModelEvaluator.kfold_cross_validation(GauClf.MVG(), DT, LT, k=3, preproc='zg', dimred=dim_red, iprint=False)
        _, mindcf_tied = BinaryModelEvaluator.kfold_cross_validation(GauClf.TiedG(), DT, LT, k=3, preproc='zg', dimred=dim_red, iprint=False)
        _, mindcf_naive = BinaryModelEvaluator.kfold_cross_validation(GauClf.NaiveBayes(), DT, LT, k=3, preproc='zg', dimred=dim_red, iprint=False)
        axs1[1, 1].plot(x, mindcf_mvg, '--o', label='Full')
        axs1[1, 1].plot(x, mindcf_tied, '--o', label='Tied')
        axs1[1, 1].plot(x, mindcf_naive, '--o', label='Naive')
        axs1[1, 1].set_ylim(0, 1)
        axs1[1, 1].set_xticks([0.1, 0.5, 0.9])
        axs1[1, 1].set_xlabel("π")
        dim_red = {'type': 'pca', 'm': 10}
        _, mindcf_mvg = BinaryModelEvaluator.kfold_cross_validation(GauClf.MVG(), DT, LT, k=3, preproc='zg', dimred=dim_red, iprint=False)
        _, mindcf_tied = BinaryModelEvaluator.kfold_cross_validation(GauClf.TiedG(), DT, LT, k=3, preproc='zg', dimred=dim_red, iprint=False)
        _, mindcf_naive = BinaryModelEvaluator.kfold_cross_validation(GauClf.NaiveBayes(), DT, LT, k=3, preproc='zg', dimred=dim_red, iprint=False)
        axs1[1, 2].plot(x, mindcf_mvg, '--o', label='Full')
        axs1[1, 2].plot(x, mindcf_tied, '--o', label='Tied')
        axs1[1, 2].plot(x, mindcf_naive, '--o', label='Naive')
        axs1[1, 2].set_ylim(0, 1)
        axs1[1, 2].set_xticks([0.1, 0.5, 0.9])
        axs1[1, 2].set_xlabel("π")
        fig1.legend(['Full', 'Tied', 'Naive'], loc='lower right')
        fig1.tight_layout()
        plt.show()

    @staticmethod
    def plot_histogramGMM(DT, LT):
        fig1, axs1 = plt.subplots(1, 3, constrained_layout = True)
        fig1.set_figheight(5)
        fig1.set_figwidth(13)
        comp = [1, 2, 4, 8, 16]
        colors = ["red", "blue", "green"]
        i = 0
        for cov in ['Full', 'Tied', 'Diag']:
            minDCF_values_01_z = []
            minDCF_values_05_z = []
            minDCF_values_09_z = []
            minDCF_values_01_zg = []
            minDCF_values_05_zg = []
            minDCF_values_09_zg = []
            for c_components in comp:
                _,minDCF_z = BinaryModelEvaluator.kfold_cross_validation(
                                                            GMMClassifier.GMM(alpha=0.1, nComponents=c_components, psi=0.01, covType=cov),
                                                            DT,
                                                            LT,
                                                            k=3,
                                                            preproc='znorm',
                                                            dimred=None,
                                                            iprint=False)
                _, minDCF_zg = BinaryModelEvaluator.kfold_cross_validation(
                                                            GMMClassifier.GMM(alpha=0.1, nComponents=c_components, psi=0.01, covType=cov),
                                                            DT,
                                                            LT,
                                                            k=3,
                                                            preproc='zg',
                                                            dimred=None,
                                                            iprint=False)
                # minDCF_values_01_z.append(minDCF_z[0])
                minDCF_values_05_z.append(minDCF_z[1])
                # minDCF_values_09_z.append(minDCF_z[2])
                # minDCF_values_01_zg.append(minDCF_zg[0])
                minDCF_values_05_zg.append(minDCF_zg[1])
                # minDCF_values_09_zg.append(minDCF_zg[2])
            ind = np.arange(len(comp))
            axs1[i].bar(x=ind, height=minDCF_values_05_z, width=0.25, alpha=0.6, color=colors[0]) # z-norm features
            axs1[i].bar(x=ind+0.25, height=minDCF_values_05_zg, width=0.25, alpha=0.6, color=colors[1]) # gau features

            axs1[i].set_xticks(ind+0.25, comp)
            axs1[i].set_xlabel("GMM components")
            axs1[i].set_title(cov)
            axs1[i].set_ylim([0, 0.6])
            i += 1
        axs1[0].set_ylabel("minDCF")
        fig1.legend(['minDCF(π = 0.5) - Z-Norm Features', 'minDCF(π = 0.5) - Gaussianization'], loc='lower right')
        plt.show()

