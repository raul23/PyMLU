"""Auto-generated module for performing data exploration.

This module is used for doing data exploration of a dataset such as computing
stats (e.g. mean, quantiles) and generating charts (e.g. bar chart and
distribution graphs) in order to better understand the dataset.
"""
from pymlu import dautils as da


def explore(config):
    # ------------------------------------
    # Data exploration: compute stats, ...
    # ------------------------------------
    data = da.DataExplorer(config=config)
    # For each column in the data (from train, test), list number of missing values
    data.count_null()
    # Compute stats (describe()) on each data (train, test)
    data.compute_stats()
    # Print first N rows from each data (train, test)
    data.head()
