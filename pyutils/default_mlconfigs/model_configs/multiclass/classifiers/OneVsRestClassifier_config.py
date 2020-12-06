from configs import config
# --------
# ML model
# --------
# One-vs-the-rest (OvR) multiclass/multilabel strategy
# ref.: https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
estimator = {
    'model_type': 'sklearn.tree.DecisionTreeClassifier',
    'model_params': {
        'max_depth': 5,
        'random_state': config.random_seed
    }
}

model = {
    'model_type': 'sklearn.multiclass.OneVsRestClassifier',
    'model_params': {
        'estimator': estimator,
    }
}
