from configs import config
# --------
# ML model
# --------
# A decision tree classifier.
# ref.: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
model = {
    'model_name': 'sklearn.tree.DecisionTreeClassifier',
    'model_params': {
        'max_depth': 5,
        'random_state': config.random_seed
    }
}
