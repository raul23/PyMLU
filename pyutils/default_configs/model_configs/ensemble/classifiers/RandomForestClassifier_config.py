from configs import config
# --------
# ML model
# --------
# A random forest classifier.
# ref.: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
model = {
    'model_type': 'sklearn.ensemble.RandomForestClassifier',
    'model_params': {
        'n_estimators': 100,
        'max_depth': 5,
        'random_state': config.random_seed
    }
}
