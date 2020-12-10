from pymlu.default_mlconfigs import config
from pymlu.mlutils import train_models

# -----------
# Main config
# -----------
main_config = {
    'custom_dataset': {
        'train_filepath': '~/Data/kaggle_datasets/titanic/train.csv',
        'test_filepath': '~/Data/kaggle_datasets/titanic/test.csv',
        'y_target': 'Survived'
    },
    'use_custom_data': True,
    'features': ["Pclass", "Sex", "SibSp", "Parch"]
}

# --------------------
# Model configurations
# --------------------
model1 = {
    'model_name': 'sklearn.ensemble.AdaBoostClassifier',
    'model_params': {
        'base_estimator': None,
        'n_estimators': 50,
        'random_state': config.random_seed
    }
}

model2 = {
    'model_name': 'sklearn.ensemble.RandomForestClassifier',
    'model_params': {
        'n_estimators': 200,
        'max_depth': 10,
        'random_state': config.random_seed
    }
}


def main():
    train_models(main_config, [model1, model2], quiet=False, logging_level='DEBUG')


if __name__ == '__main__':
    main()
