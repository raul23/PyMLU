"""Data analysis utilities
"""
import logging.config
from logging import NullHandler

# import ipdb
import pandas as pd

from pyutils import genutils as ge

logger = logging.getLogger(ge.get_short_logger_name(__name__))
logger.addHandler(NullHandler())

logger_data = logging.getLogger('data')
logger_data.addHandler(NullHandler())


class DataExplorer:
    def __init__(self, data_filepath=None, train_filepath=None,
                 valid_filepath=None, test_filepath=None, data_stats=True,
                 train_stats=True, valid_stats=True, test_stats=True,
                 excluded_cols=None, data_head=5, train_head=5,
                 valid_head=5, test_head=5, data_isnull=True,
                 train_isnull=True, valid_isnull=True, test_isnull=True,
                 *args, **kwargs):
        self.data_filepath = data_filepath
        self.train_filepath = train_filepath
        self.valid_filepath = valid_filepath
        self.test_filepath = test_filepath
        self.data_stats = data_stats
        self.train_stats = train_stats
        self.valid_stats = valid_stats
        self.test_stats = test_stats
        self.excluded_cols = excluded_cols
        self.data_head = data_head
        self.train_head = train_head
        self.valid_head = valid_head
        self.test_head = test_head
        self.data_isnull = data_isnull
        self.train_isnull = train_isnull
        self.valid_isnull = valid_isnull
        self.test_isnull = test_isnull
        # ---------
        # Load data
        # ---------
        logger.info("Loading data")
        self.datasets = self._load_data()

    def compute_stats(self):
        for data_type, data in self.datasets.items():
            if data is not None \
                    and self.__getattribute__('{}_stats'.format(data_type)):
                compute_stats(data, data_type,
                              excluded_cols=self.excluded_cols)

    def count_null(self):
        for data_type, data in self.datasets.items():
            if data is not None \
                    and self.__getattribute__('{}_isnull'.format(data_type)):
                logger_data.info(
                    "*** Number of missing values for each feature in {} "
                    "***\n{}\n".format(
                        data_type,
                        data.isnull().sum()))

    def head(self):
        for data_type, data in self.datasets.items():
            n_rows = self.__getattribute__('{}_head'.format(data_type))
            if data is not None and n_rows:
                logger_data.info("*** First {} rows of {} ***\n{}\n".format(
                    n_rows,
                    data_type,
                    data.head(n_rows)))

    def _load_data(self):
        datasets = {
            'data': None,
            'train': None,
            'valid': None,
            'test': None
        }
        for data_type in datasets.keys():
            filepath = self.__getattribute__('{}_filepath'.format(data_type))
            if filepath:
                datasets[data_type] = pd.read_csv(filepath)
            else:
                # TODO: logging (no filepath for dataset)
                pass
        return datasets


def remove_columns(data, excluded_cols):
    cols = set(data.columns)
    valid_cols = cols - set(excluded_cols)
    if excluded_cols:
        logger_data.info(f"Excluded columns: {cols.intersection(excluded_cols)}")
    return data[valid_cols]


def remove_strings_from_cols(data):
    num_cols = []
    strings_cols = []
    for col, dtype in data.dtypes.to_dict().items():
        if dtype.name == 'object':
            strings_cols.append(col)
        else:
            num_cols.append(col)
    if num_cols:
        logger_data.info(f"Rejected strings columns: {strings_cols}")
        return data[num_cols]
    else:
        logger_data.info("No strings columns found in the data")
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
    logger_data.info(data.describe())
    logger_data.info("")
