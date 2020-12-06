from configs import config
# --------
# ML model
# --------
# - Linear classifiers (SVM, logistic regression, etc.) with SGD training.
# - Always scale the input.
# ref.: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
model = {
    'model_type': 'sklearn.linear_model.SGDClassifier',
    'model_params': {
        'max_iter': 1000,
        'tol': 1e-3,
        # Used for shuffling the data, when shuffle is set to True. Pass an int
        # for reproducible output across multiple function calls.
        'random_state': config.random_seed
    },
    'scale_input': True
}
