import numpy as np
from sklearn.metrics import f1_score


def f1_score_(actual, predicted, average='macro'):
    actual = np.array(actual)
    predicted = np.array(predicted)
    return f1_score_(actual, predicted, average=average)
