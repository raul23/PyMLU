# --------
# ML model
# --------
# - Similar to SVC but uses a parameter to control the number of support vectors.
# - No param C (reg)
# - Example makes use of StandardScaler
# ref.: https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html#sklearn.svm.NuSVC
model = {
    'model_type': 'sklearn.svm.NuSVC',
    'model_params': {
        'nu': 0.5,  # control the number of support vectors
        'kernel': 'rbf',  # {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}
        'degree': 3,  # if kernel == 'poly'
        'gamma': 'scale',  # if kernel in ['poly', 'rbf', 'sigmoid']
        'tol': 1e-3,
        'random_state': 1
    },
    'scale_input': True
}
