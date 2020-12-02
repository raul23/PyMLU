from configs import config
# --------
# ML model
# --------
# - Probability calibration with isotonic regression or logistic regression.
# ref.: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
base_model = {
    'model_type': 'sklearn.tree.DecisionTreeClassifier',
    'model_params': {
        'max_depth': 5,
        'random_state': config.random_seed
    }
}

model = {
    'model_type': 'sklearn.calibration.CalibratedClassifierCV',
    'model_params': {
        'base_estimator': base_model
    }
}
