# ---------------
# General options
# ---------------
quiet = False
verbose = False

# --------------
# Data filepaths
# --------------
data_filepath = ''
train_filepath = ''
valid_filepath = ''
test_filepath = ''
y_target = ''

# ------------------
# Data preprocessing
# ------------------
features = ["Pclass", "Sex", "SibSp", "Parch"]
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
        'random_state': 1
    }
}
