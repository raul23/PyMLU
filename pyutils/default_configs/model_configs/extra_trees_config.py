scale_input = False
# --------
# ML model
# --------
# An extra-trees classifier.
# ref.: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
model = {
    'model_type': 'sklearn.ensemble.ExtraTreesClassifier',
    'model_params': {
        'n_estimators': 100,
        'random_state': 1
    }
}
