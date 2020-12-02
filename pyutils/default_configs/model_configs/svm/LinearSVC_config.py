from configs import config
# --------
# ML model
# --------
# Example makes use of StandardScaler
# ref.: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
model = {
    'model_type': 'sklearn.svm.LinearSVC',
    'model_params': {
        'loss': 'squared_hinge',  # hinge
        'tol': 1e-4,
        'C': 1,  # C > 0
        'random_state': config.random_seed
    },
    'scale_input': True
}
