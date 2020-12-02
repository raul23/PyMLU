"""Exploring the Kaggle's Titanic dataset

This module only does data exploration of the Titanic dataset such as computing
stats (e.g. mean, quantiles) and generating charts (e.g. bar chart and
distribution graphs) in order to better understand the dataset.

Thus after having explored the dataset under various aspects, we can use
machine learning (ML) models to predict who will survive based on the Titanic
passenger data (e.g. name, age, price of ticket, etc).

The script :mod:`train_model` is used for training the ML model as defined in
:mod:`config`.

Dataset website: https://www.kaggle.com/c/titanic
"""
import logging.config
from logging import NullHandler

from pyutils import dautils as da
from pyutils import genutils as ge

logger = logging.getLogger(__name__)
logger.addHandler(NullHandler())


def main():
    global logger
    bp = ge.ConfigBoilerplate(__file__)
    logger = bp.get_logger()

    data = da.DataExplorer(**bp.get_cfg_dict())
    data.count_null()
    data.compute_stats()
    data.head()


if __name__ == '__main__':
    main()
