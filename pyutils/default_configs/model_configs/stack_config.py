# --------
# ML model
# --------
estimator_01 = {
    'model_type': 'sklearn.ensemble.RandomForestClassifier',
    'model_params': {
        'n_estimators': 100,
        'max_depth': 5,
        'random_state': 1
    }
}

estimator_02 = {
    'model_type': 'sklearn.svm.LinearSVC',
    'model_params': {
        'loss': 'squared_hinge',
        'tol': 1e-4,
        'C': 1,
        'random_state': 1,
        'max_iter': 2000
    },
    'scale_input': True
}

final_estimator = {
    'model_type': 'sklearn.linear_model.LogisticRegression',
    'model_params': {
        'random_state': 1
    }
}

model = {
    'model_type': 'sklearn.ensemble.StackingClassifier',
    'model_params': {
        'estimators': [estimator_01, estimator_02],
        'final_estimator': final_estimator
    }
}
