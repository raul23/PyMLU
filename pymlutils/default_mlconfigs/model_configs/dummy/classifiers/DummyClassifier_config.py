from configs import config
# --------
# ML model
# --------
# - DummyClassifier is a classifier that makes predictions using simple rules.
# - Do not use it for real problems
# ref.: https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
model = {
    'model_type': 'sklearn.dummy.DummyClassifier',
    'model_params': {
        # stratified, most_frequent, prior, uniform, constant
        'strategy': 'constant',
        'random_state': config.random_seed,
        # Sseful only for the “constant” strategy.
        'constant': 0
    }
}
