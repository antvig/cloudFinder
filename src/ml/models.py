import numpy as np

class VotingClassifier(object):
    """Stripped-down version of VotingClassifier that uses prefit estimators"""

    def __init__(self, voting="hard", weights=None):
        self.estimators = None
        self.named_estimators = None
        self.voting = voting
        self.weights = weights

    def fit_from_base_estimators(self, X, y, sample_weight=None):
        raise NotImplementedError

    def fit_predict_cv(self, X, y, cv_split, base_estimator):
        """ Fit the voting classifier from cross validation. The estimators are the models learnt for each train set of the CV.
        The prediction are computed on the test set of the CV. 
        """

        y_pred = np.zeros((len(y), 3))
        estimators = []
        c = 1
        for train_index, test_index in cv_split:
            print("iteration #{}".format(c))
            estimator = base_estimator
            print("  -> fit")
            estimator.fit(X[train_index], y[train_index])
            print("  -> predict")
            y_pred[test_index, 0] = estimator.predict(X[test_index])
            y_pred[test_index, 1] = estimator.predict_proba(X[test_index])[
                :, 1
            ]
            y_pred[test_index, 2] = c
            estimators.append((c, estimator))
            c += 1

        self._add_estimators(estimators)

        return y_pred

    def from_fitted_estimators(cls, estimators, voting="hard", weights=None):
        """ Create a voting classifier from fitted estimators
        """
        vc = cls(voting, weights)
        vc._add_estimators(estimators)

        return vc

    def predict(self, X):
        """ Predict class labels for X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.
        """

        check_is_fitted(self, "estimators")
        if self.voting == "soft":
            maj = np.argmax(self.predict_proba(X), axis=1)

        else:  # 'hard' voting
            predictions = self._predict(X)
            maj = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                axis=1,
                arr=predictions.astype("int"),
            )
        return maj

    def _collect_probas(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict_proba(X) for clf in self.estimators])

    def _predict_proba(self, X):
        """Predict class probabilities for X in 'soft' voting """
        if self.voting == "hard":
            raise AttributeError(
                "predict_proba is not available when" " voting=%r" % self.voting
            )
        check_is_fitted(self, "estimators")
        avg = np.average(self._collect_probas(X), axis=0, weights=self.weights)
        return avg

    @property
    def predict_proba(self):
        """Compute probabilities of possible outcomes for samples in X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        ----------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        """
        return self._predict_proba

    def transform(self, X):
        """Return class labels or probabilities for X for each estimator.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        -------
        If `voting='soft'`:
          array-like = [n_classifiers, n_samples, n_classes]
            Class probabilities calculated by each classifier.
        If `voting='hard'`:
          array-like = [n_samples, n_classifiers]
            Class labels predicted by each classifier.
        """
        check_is_fitted(self, "estimators")
        if self.voting == "soft":
            return self._collect_probas(X)
        else:
            return self._predict(X)

    def _predict(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.estimators]).T

    def _add_estimators(self, estimators):
        self.estimators = [e[1] for e in estimators]
        self.named_estimators = dict(estimators)
