from configs import config
# --------
# ML model
# --------
# An extremely randomized tree classifier.
# Warning: Extra-trees should only be used within ensemble methods.
# ref.: https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html
base_model = {
    'model_name': 'sklearn.tree.ExtraTreeClassifier',
    'model_params': {
        'max_depth': 5,
        'random_state': 1
    }
}
model = {
    'model_name': 'sklearn.ensemble.BaggingClassifier',
    'model_params': {
        'base_estimator': base_model,
        'n_estimators': 50,
        'random_state': config.random_seed
    }
}
