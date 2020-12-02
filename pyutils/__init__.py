"""TODO
"""
# For debugging purposes
__test_version__ = "0.0.0a0"
# Version of package
__version__ = "0.1.0a0"

CONFIGS_DIRNAME = 'configs'
MODEL_CONFIGS_DIRNAME = 'model_configs'
MODEL_FNAME_ENDING = '_config.py'
SKLEARN_MODULES = ['calibration', 'dummy', 'ensemble', 'gaussian_process',
                   'linear_model', 'multiclass', 'multioutput', 'naive_bayes',
                   'neighbors', 'neural_network', 'semi_supervised', 'svm',
                   'tree']
SKLEARN_MODULES.sort()