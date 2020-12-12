from pymlu.dautils import explore_data

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


def main():
    explore_data(main_config, quiet=False, logging_level='INFO',
                 logging_formatter='only_msg')


if __name__ == '__main__':
    main()
