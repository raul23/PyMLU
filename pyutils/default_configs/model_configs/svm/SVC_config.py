from configs import config
# --------
# ML model
# --------
# Example makes use of StandardScaler
# ref.: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
model = {
    'model_type': 'sklearn.svm.SVC',
    'model_params': {
        'C': 1,  # C > 0
        'kernel': 'rbf',  # {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}
        'degree': 3,  # if kernel == 'poly'
        'gamma': 'scale',  # if kernel in ['poly', 'rbf', 'sigmoid']
        'tol': 1e-3,
        'random_state': config.random_seed
    },
    'scale_input': True
}
