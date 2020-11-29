# --------
# ML model
# --------
# - A Bagging classifier.
# - base_estimator: if None, then the base estimator is a decision tree.
# ref.: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
model = {
    'model_type': 'sklearn.ensemble.BaggingClassifier',
    'model_params': {
        'base_estimator': None,
        'n_estimators': 10,
        'random_state': 1
    }
}
