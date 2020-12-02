from configs import config
# --------
# ML model
# --------
model = {
    'model_type': 'sklearn.linear_model.LogisticRegression',
    'model_params': {
        'random_state': config.random_seed
    }
}
