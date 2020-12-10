from configs import config
# --------
# ML model
# --------
# - An AdaBoost classifier.
# - base_estimator: if None, then the base estimator is a decision tree.
# ref.: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
model = {
    'model_name': 'sklearn.ensemble.AdaBoostClassifier',
    'model_params': {
        'base_estimator': None,
        'n_estimators': 50,
        'random_state': config.random_seed
    }
}
