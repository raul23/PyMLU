"""TODO Auto-generated script
"""
import logging.config
from logging import NullHandler

from sklearn.metrics import accuracy_score

from pyutils import genutils as ge, mlutils as ml

logger = logging.getLogger(ge.get_logger_name(__name__, __file__, 'scripts'))
logger.addHandler(NullHandler())

logger_data = logging.getLogger('data')
logger_data.addHandler(NullHandler())


def main():
    # ---------------------------------
    # Setup logging and get config dict
    # ---------------------------------
    cfg_dicts = ge.ConfigBoilerplate(__file__).get_cfg_dict()

    # ---------
    # Load data
    # ---------
    data = ml.Dataset(**cfg_dicts[0])

    # For each model, get its config dict
    for i, cfg_dict in enumerate(cfg_dicts, start=1):
        # ---------
        # Get model
        # ---------
        logger_data.info("")
        logger.info(f"Model #{i}: {cfg_dict['model']['model_type']}")
        clf = ml.get_model(**cfg_dict['model'])

        # -----------
        # Train model
        # -----------
        logger.info("Training model")
        logger_data.debug(f"{clf}")
        clf.fit(data.X_train, data.y_train)
        score = clf.score(data.X_train, data.y_train)
        logger.info(f"Score on train: {score}")

        # -----------------------
        # Get predictions on test
        # -----------------------
        logger.info("Getting predictions from test data")
        predictions = clf.predict(data.X_test)
        if data.y_test is not None:
            test_score = accuracy_score(data.y_test, predictions)
            logger.info(f"Score on test: {test_score}")


if __name__ == '__main__':
    main()
