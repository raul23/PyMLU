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
    'data_prop': (0.7, 0.3),
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
train_stats= True
valid_stats= True
test_stats= True
# Excluded columns can be names of features (column names) or indices (column positions)
# e.g. excluded_cols = ['PassengerId']
#      excluded_cols = [4]
excluded_cols= None

# ------------------
# HEAD: first N rows
# ------------------
train_head = 5
valid_head = 5
test_head = 5

# ------------------------------
# Count number of missing values
# ------------------------------
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
