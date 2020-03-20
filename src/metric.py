from sklearn import metrics as skmetrics


class ClasificationMetric:
    def __init__(self):
        self.metrics = {
            "accuracy" : self._accuracy,
            "recall" : self._recall,
            "precision":self._precision,
            "f1":self._f1,
            "auc":self._auc,
            "logloss":self._logloss
        }

    def __call__(self,metric,y_true,y_pred,y_prob = None):
        if metric not in self.metrics:
            raise Exception("we dont offer this metrics right now")
        
        if metric == "auc":
            if y_prob == None:
                raise Exception("y_prob cannot be none")
            else:
                return self._auc(y_true = y_true,y_pred = y_prob)

        elif metric == "logloss":
            if y_prob == None:
                raise Exception("y_prob cannot be none")
            else:
                return self._logloss(y_true = y_true,y_pred = y_prob)

        return self.metrics[metric](y_true = y_true, y_pred = y_pred)

    @staticmethod
    def _accuracy(y_true,y_pred):
        return skmetrics.accuracy_score(y_true = y_true,y_pred = y_pred)

    @staticmethod
    def _recall(y_true,y_pred):
        return skmetrics.recall_score(y_true = y_true,y_pred = y_pred)

    @staticmethod
    def _precision(y_true,y_pred):
        return skmetrics.precision_score(y_true = y_true,y_pred = y_pred)

    @staticmethod
    def _f1(y_true,y_pred):
        return skmetrics.f1_score(y_true = y_true,y_pred = y_pred)

    @staticmethod
    def _auc(y_true,y_pred):
        return skmetrics.roc_auc_score(y_true = y_true,y_score = y_pred) ## it should be prob

    @staticmethod
    def _logloss(y_true,y_pred):
        return skmetrics.log_loss(y_true = y_true,y_pred = y_pred) ## it should be prob

    



