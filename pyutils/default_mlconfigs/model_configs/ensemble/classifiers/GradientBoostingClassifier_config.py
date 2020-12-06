from configs import config
# --------
# ML model
# --------
# Gradient Boosting for classification.
# ref.: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
model = {
    'model_type': 'sklearn.ensemble.GradientBoostingClassifier',
    'model_params': {
        'n_estimators': 100,
        'random_state': config.random_seed
    }
}
