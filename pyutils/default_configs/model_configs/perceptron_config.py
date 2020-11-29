# --------
# ML model
# --------
# Perceptron() is equivalent to SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None).
# ref.: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html
model = {
    'model_type': 'sklearn.linear_model.Perceptron',
    'model_params': {
        'random_state': 1
    },
    'scale_input': True
}
