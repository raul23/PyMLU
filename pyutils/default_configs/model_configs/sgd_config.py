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
        # Ex: if random_state = 0 -> 0.7912457912457912
        #     if random_state = 1 -> 0.7575757575757576
        'random_state': 0
    },
    'scale_input': True
}
