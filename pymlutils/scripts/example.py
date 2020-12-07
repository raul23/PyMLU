from pymlutils import genutils as ge
from pymlutils.default_mlconfigs import config
from pymlutils.default_mlmodules import train_models

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
    'model_type': 'sklearn.ensemble.AdaBoostClassifier',
    'model_params': {
        'base_estimator': None,
        'n_estimators': 50,
        'random_state': config.random_seed
    }
}

model2 = {
    'model_type': 'sklearn.ensemble.RandomForestClassifier',
    'model_params': {
        'n_estimators': 200,
        'max_depth': 10,
        'random_state': config.random_seed
    }
}


def main():
    # Update default config dict with dataset and model configs
    configs = ge.get_configs(main_config, [model1, model2], quiet=False,
                             logging_level='INFO')
    train_models.train(configs)


if __name__ == '__main__':
    main()
