"""TODO
"""
# For debugging purposes
__test_version__ = "0.0.0a0"
# Version of package
__version__ = "0.1.0a0"


# TODO (IMPORTANT): should be in genutils
CONFIGS_DIRNAME = 'mlconfigs'
MODEL_CONFIGS_DIRNAME = 'model_configs'
MODEL_FNAME_SUFFIX = '_config.py'
MODULES_DIRNAME = 'mlmodules'
# TODO (IMPORTANT): should be in mlutils
# TODO: get categories from directory
SKLEARN_MODULES = ['calibration', 'dummy', 'ensemble', 'gaussian_process',
                   'linear_model', 'multiclass', 'multioutput', 'naive_bayes',
                   'neighbors', 'neural_network', 'semi_supervised', 'svm',
                   'tree']
SKLEARN_MODULES.sort()