"""Auto-generated script for performing data exploration.

This module is used for doing data exploration of a dataset such as computing
stats (e.g. mean, quantiles) and generating charts (e.g. bar chart and
distribution graphs) in order to better understand the dataset.

The script :mod:`train_model` is used for training the ML model as defined in
:mod:`configs`.
"""
import logging.config
from logging import NullHandler

from pyutils import dautils as da
from pyutils import genutils as ge

logger = logging.getLogger(ge.get_logger_name(__name__, __file__, 'scripts'))
logger.addHandler(NullHandler())


def main():
    # ---------------------------------
    # Setup logging and get config dict
    # ---------------------------------
    cfg_dict = ge.ConfigBoilerplate(__file__).get_cfg_dict()

    # ------------------------------------
    # Data exploration: compute stats, ...
    # ------------------------------------
    data = da.DataExplorer(**cfg_dict)
    # For each column in the data (from train, test), list number of missing values
    data.count_null()
    # Compute stats (describe()) on each data (train, test)
    data.compute_stats()
    # Print first N rows from each data (train, test)
    data.head()


if __name__ == '__main__':
    main()