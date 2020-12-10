from configs import config
# --------
# ML model
# --------
estimator_01 = {
    'model_name': 'sklearn.ensemble.RandomForestClassifier',
    'model_params': {
        'n_estimators': 100,
        'max_depth': 5,
        'random_state': config.random_seed
    }
}

estimator_02 = {
    'model_name': 'sklearn.svm.LinearSVC',
    'model_params': {
        'loss': 'squared_hinge',
        'tol': 1e-4,
        'C': 1,
        'random_state': config.random_seed,
        'max_iter': 2000
    },
    'scale_input': True
}

final_estimator = {
    'model_name': 'sklearn.linear_model.LogisticRegression',
    'model_params': {
        'random_state': config.random_seed
    }
}

model = {
    'model_name': 'sklearn.ensemble.StackingClassifier',
    'model_params': {
        'estimators': [estimator_01, estimator_02],
        'final_estimator': final_estimator
    }
}
