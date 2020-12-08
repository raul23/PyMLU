"""Auto-generated module for performing data exploration.

This module is used for doing data exploration of a dataset such as computing
stats (e.g. mean, quantiles) and generating charts (e.g. bar chart and
distribution graphs).
"""
from pymlutils import dautils as da, genutils as ge

logger = ge.init_log(__name__)
logger_data = ge.init_log('data')


def explore(cfg_dict):
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
