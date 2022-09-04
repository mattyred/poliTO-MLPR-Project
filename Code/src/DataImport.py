import numpy as np
import Constants as CONST


def read_data_training(path):
    DT = np.zeros(shape=(CONST.NFEATURES, CONST.NTSAMPLES), dtype='float32')  # DT: Data Training
    LT = np.zeros(CONST.NTSAMPLES, dtype='int32')  # LT: Labels Training
    with open(path, 'r') as file:
        i = 0
        for line in file:
            features_list = line.split(',')
            features = np.array(features_list[0:CONST.NFEATURES], dtype='float32').reshape(-1, 1)
            label = int(features_list[-1])
            DT[:, i:i + 1] = features
            LT[i] = label
            i += 1
    return DT, LT

def read_data_evaluation(path):
    DE = np.zeros(shape=(CONST.NFEATURES, CONST.NESAMPLES), dtype='float32')  # DT: Data Training
    LE = np.zeros(CONST.NESAMPLES, dtype='int32')  # LT: Labels Training
    with open(path, 'r') as file:
        i = 0
        for line in file:
            features_list = line.split(',')
            features = np.array(features_list[0:CONST.NFEATURES], dtype='float32').reshape(-1, 1)
            label = int(features_list[-1])
            DE[:, i:i + 1] = features
            LE[i] = label
            i += 1
    return DE, LE


def read_test(path):
    pass
