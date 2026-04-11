from abc import ABC, abstractmethod
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
)


class Measure(ABC):
    @abstractmethod
    def evaluate(self, y_true, y_pred, y_score):
        pass


class Recall(Measure):
    def evaluate(self, y_true, y_pred, y_score):
        return recall_score(y_true, y_pred)


class Precision(Measure):
    def evaluate(self, y_true, y_pred, y_score):
        return precision_score(y_true, y_pred)


class F1(Measure):
    def evaluate(self, y_true, y_pred, y_score):
        return f1_score(y_true, y_pred)


class BalancedAccuracy(Measure):
    def evaluate(self, y_true, y_pred, y_score):
        return balanced_accuracy_score(y_true, y_pred)


class RocAuc(Measure):
    def evaluate(self, y_true, y_pred, y_score):
        return roc_auc_score(y_true, y_score)


class PRAuc(Measure):
    def evaluate(self, y_true, y_pred, y_score):
        return average_precision_score(y_true, y_score)
