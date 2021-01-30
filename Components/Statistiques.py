"""
Author : Chayma Zatout
"""
from sklearn import metrics

class Statistiques:

    @staticmethod
    def precision(y_true, y_pred):
        return metrics.precision_score(y_true, y_pred)

    @staticmethod
    def recall(y_true, y_pred):
        return metrics.recall_score(y_true, y_pred)

    @staticmethod
    def accuracy(y_true, y_pred):
        return metrics.accuracy_score(y_true, y_pred)

    @staticmethod
    def f_measure(y_true, y_pred):
        return metrics.f1_score(y_true, y_pred)

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        return metrics.confusion_matrix(y_true, y_pred)

    @staticmethod
    def roc(y_true, y_pred):
        pass
