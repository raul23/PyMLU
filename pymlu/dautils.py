"""Data analysis utilities
"""
import copy
import logging
from logging import NullHandler

from pymlu.default_mlmodules.explore_data import explore
from pymlu.genutils import get_config_from_locals, get_configs
from pymlu.mlutils import Dataset

pandas = None

logger = logging.getLogger(__name__)
logger.addHandler(NullHandler())

logger_data = logging.getLogger('data')
logger_data.addHandler(NullHandler())


class DataExplorer:
    def __init__(self, builtin_dataset=None, custom_dataset=None,
                 use_custom_data=False,
                 train_stats=True, valid_stats=True, test_stats=True,
                 excluded_cols=None, data_head=5, train_head=5,
                 valid_head=5, test_head=5, data_isnull=True,
                 train_isnull=True, valid_isnull=True, test_isnull=True,
                 config=None, *args, **kwargs):
        cfg = get_config_from_locals(config, locals(), ignored_keys=['config'])
        global pandas
        logger.debug("Importing pandas...")
        # Lazy import
        import pandas
        self.builtin_dataset = cfg.builtin_dataset
        self.custom_dataset = cfg.custom_dataset
        self.use_custom_data = cfg.use_custom_data
        self.train_stats = cfg.train_stats
        self.valid_stats = cfg.valid_stats
        self.test_stats = cfg.test_stats
        self.excluded_cols = cfg.excluded_cols
        self.train_head = cfg.train_head
        self.valid_head = cfg.valid_head
        self.test_head = cfg.test_head
        self.train_isnull = cfg.train_isnull
        self.valid_isnull = cfg.valid_isnull
        self.test_isnull = cfg.test_isnull
        # ---------
        # Load data
        # ---------
        logger.info("Loading dataset")
        self.dataset = Dataset(cfg.builtin_dataset, cfg.custom_dataset,
                               cfg.use_custom_data)
        logger_data.info("")

    def compute_stats(self):
        for data_type in self.dataset.data_types:
            X_data, y_data = self.dataset.get_data(data_type)
            if (X_data is not None or y_data is not None) and \
                    self.__getattribute__('{}_stats'.format(data_type)):
                # TODO: explain why we do concat()
                concat_data = pandas.concat([X_data, y_data], axis=1)
                compute_stats(concat_data, data_type,
                              excluded_cols=self.excluded_cols)
            else:
                # TODO: log (missing data)
                pass

    def count_null(self):
        for data_type in self.dataset.data_types:
            X_data, y_data = self.dataset.get_data(data_type)
            if (X_data is not None or y_data is not None) and \
                    self.__getattribute__('{}_isnull'.format(data_type)):
                concat_data = pandas.concat([X_data, y_data], axis=1)
                logger_data.info(
                    "*** Number of missing values for each column in {} "
                    "***\n{}\n".format(
                        data_type,
                        concat_data.isnull().sum()))
            else:
                # TODO: log (missing data)
                pass

    def head(self):
        for data_type in self.dataset.data_types:
            n_rows = self.__getattribute__('{}_head'.format(data_type))
            X_data, y_data = self.dataset.get_data(data_type)
            if (X_data is not None or y_data is not None) and n_rows:
                concat_data = pandas.concat([X_data, y_data], axis=1)
                logger_data.info("*** First {} rows of {} ***\n{}\n".format(
                    n_rows,
                    data_type,
                    concat_data.head(n_rows)))
            else:
                # TODO: log (missing data)
                pass


def remove_columns(data, excluded_cols):
    if isinstance(excluded_cols[0], str):
        all_cols = data.columns
        valid_cols = all_cols.to_list()
        invalid_cols = []
        excluded_cols_copy = copy.copy(excluded_cols)
        for col in excluded_cols:
            if col in all_cols:
                valid_cols.remove(col)
            else:
                invalid_cols.append(col)
                excluded_cols_copy.remove(col)
        if excluded_cols_copy:
            logger_data.info(f"Excluded columns: {excluded_cols_copy}")
        if invalid_cols:
            logger_data.info(f"Invalid excluded columns: {invalid_cols}")
        data = data[valid_cols]
    else:
        col_indices = [i for i in range(0, data.shape[1])]
        valid_col_indices = set(col_indices) - set(excluded_cols)
        excluded_cols_copy = list(set(col_indices) - valid_col_indices)
        if excluded_cols_copy:
            excluded_cols = data.columns[list(excluded_cols_copy)].to_list()
            logger_data.info(f"Excluded columns: {excluded_cols}")
        data = data.iloc[:, list(valid_col_indices)]
    return data


def remove_strings_from_cols(data):
    num_cols = []
    strings_cols = []
    for col, dtype in data.dtypes.to_dict().items():
        if dtype.name == 'object':
            strings_cols.append(col)
        else:
            num_cols.append(col)
    if strings_cols:
        logger_data.info(f"Rejected string columns: {strings_cols}")
        # If only string columns, empty dataset returned
        return data[num_cols]
    else:
        logger_data.info("No string columns found in the data")
        return data


def compute_stats(data, name='data', include_strings=False,
                         excluded_cols=None):
    first_msg = f"*** Stats for {name} ***"
    logger_data.info("*" * len(first_msg))
    logger_data.info(f"{first_msg}")
    logger_data.info("*" * len(first_msg))
    logger_data.info(f"Shape: {data.shape}")
    if excluded_cols:
        data = remove_columns(data, excluded_cols)
    if not include_strings:
        data = remove_strings_from_cols(data)
    logger_data.info("")
    if len(data.columns):
        logger_data.info(data.describe())
    else:
        logger.warning("All the data columns were removed! Skipping computing "
                       f"stats for {name}.")
    logger_data.info("")


def explore_data(main_config_dict=None, model_configs=None, quiet=False,
                 verbose=False, logging_level=None, logging_formatter=None):
    explore(get_configs(**locals())[0])
