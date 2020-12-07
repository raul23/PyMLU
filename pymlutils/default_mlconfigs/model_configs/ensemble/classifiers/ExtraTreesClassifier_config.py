from configs import config
# --------
# ML model
# --------
# An extra-trees classifier.
# ref.: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
model = {
    'model_name': 'sklearn.ensemble.ExtraTreesClassifier',
    'model_params': {
        'n_estimators': 100,
        'random_state': config.random_seed
    }
}
