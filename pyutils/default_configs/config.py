# ---------------
# General options
# ---------------
quiet = False
verbose = False
random_seed = 1

# ----------------
# Built-in dataset
# ----------------
builtin_dataset = {
    'name': 'iris',
    # (train_proportion, test_proportion)
    'data_prop': (0.9, 0.1),
    'shuffle_data': True
}

# --------------
# Custom dataset
# --------------
# TODO: data_filepath and valid_filepath
custom_dataset = {
    'train_filepath': None,
    'test_filepath': None,
    'y_target': None
}
use_custom_data = False

# ------------------
# Data preprocessing
# ------------------
# Features can be names of features (column names) or indices (column positions)
# e.g. features = ['featureA', 'featureB']
#      features = [i for i in range(0, 10)]
features = None
# One-hot encode the data using pandas get_dummies()
get_dummies = True
# TODO: implement this option, scale input for all models
# scale_input = False

# -------------
# Compute stats
# -------------
data_stats= True
train_stats= True
valid_stats= True
test_stats= True
excluded_cols= ['PassengerId']

# ------------------
# HEAD: first N rows
# ------------------
data_head = 5
train_head = 5
valid_head = 5
test_head = 5

# ------------------------------
# Count number of missing values
# ------------------------------
data_isnull = True
train_isnull = True
valid_isnull = True
test_isnull = True

# --------
# ML model
# --------
model = {
    'model_type': 'sklearn.ensemble.RandomForestClassifier',
    'model_params': {
        'n_estimators': 100,
        'max_depth': 5,
        'random_state': random_seed
    }
}
