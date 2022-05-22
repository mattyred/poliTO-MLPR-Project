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