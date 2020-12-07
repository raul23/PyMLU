"""Auto-generated module for training models.
TODO
"""
from pymlutils import genutils as ge, mlutils as ml

logger = ge.init_log(__name__)
logger_data = ge.init_log('data')


def train(configs):
    # ---------
    # Load data
    # ---------
    data = ml.Dataset(**configs[0])

    # For each model, get its config dict
    for i, cfg_dict in enumerate(configs, start=1):
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
            # Slow to import
            from sklearn.metrics import accuracy_score

            test_score = accuracy_score(data.y_test, predictions)
            logger.info(f"Score on test: {test_score}")
