from configs import config
# --------
# ML model
# --------
# Passive Aggressive Classifier
# ref.: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html
model = {
    'model_type': 'sklearn.linear_model.PassiveAggressiveClassifier',
    'model_params': {
        'max_iter': 1000,
        'random_state': config.random_seed
    }
}
