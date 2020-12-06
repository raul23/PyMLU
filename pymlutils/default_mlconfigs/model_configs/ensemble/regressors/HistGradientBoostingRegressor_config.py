from configs import config
# --------
# ML model
# --------
# - Histogram-based Gradient Boosting Classification Tree.
# - This estimator is much faster than GradientBoostingClassifier for big
#   datasets (n_samples >= 10 000).
#
# Note: this estimator is still experimental for now: To use it, you need to
#       explicitly import enable_hist_gradient_boosting
#
# ref.: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
model = {
    'model_type': 'sklearn.ensemble.HistGradientBoostingClassifier',
    'model_params': {
        'learning_rate': 0.1,  # 1 for no shrinkage
        'random_state': config.random_seed
    }
}
