import numpy as np


class BinaryModelEvaluator:

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

    @staticmethod
    def accuracy(predicted_labels, actual_labels):
        return np.sum([predicted_labels == actual_labels]) / len(predicted_labels)

    @staticmethod
    def error_rate(predicted_labels, actual_labels):
        return 1 - np.sum([predicted_labels == actual_labels]) / len(predicted_labels)

    @staticmethod
    def kfold_cross_validation(model=None, D=None, L=None, k=10):
        # apply cross validation on D with k-1 folds for training and 1 fold for testing
        Nsamples = D.shape[1]
        np.random.seed(0)
        idx = np.random.permutation(Nsamples)
        folds = np.array_split(idx, k)
        avg_error_rate = 0
        for i in range(k):
            # 1 Test fold
            fold_test = folds[i]
            Dtest = D[:, fold_test]
            Ltest = L[fold_test]
            # k-1 Train folds
            folds_train = []
            for j in range(k):
                if j != i:
                    folds_train.append(folds[j])
            Dtrain = D[:, np.array(folds_train).flat]
            Ltrain = L[np.array(folds_train).flat]
            labels = model.train(Dtrain, Ltrain).predict(Dtest)
            avg_error_rate += BinaryModelEvaluator.error_rate(labels, Ltest)
        avg_error_rate /= k
        print('Average error rate: %.2f%%' % (avg_error_rate*100))

