# --------
# ML model
# --------
# Gaussian process classification (GPC) based on Laplace approximation.
# ref.: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html
model = {
    'model_type': 'sklearn.gaussian_process.GaussianProcessClassifier',
    'model_params': {
        # If None is passed, the kernel “1.0 * RBF(1.0)” is used as default
        'kernel': None,
        'random_state': 1
    }
}
