from configs import config
# --------
# ML model
# --------
# Classifier using Ridge regression.
# ref.: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html
model = {
    'model_type': 'sklearn.linear_model.RidgeClassifier',
    'model_params': {
        'alpha': 1.0,
        'random_state': config.random_seed
    }
}
