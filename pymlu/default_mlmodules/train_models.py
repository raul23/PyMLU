"""Auto-generated module for training models.
TODO
"""
from pymlu import genutils as ge, mlutils as ml

logger = ge.init_log(__name__)
logger_data = ge.init_log('data')


def train(configs):
    # ---------
    # Load data
    # ---------
    data = ml.Dataset(config=configs[0])

    # For each model, get its config dict
    for i, model_cfg in enumerate(configs, start=1):
        # ---------
        # Get model
        # ---------
        logger_data.info("")
        logger.info(f"Training model #{i}: {model_cfg.model.model_name}")
        clf = ml.get_model(model_config=model_cfg.model)

        # -----------
        # Train model
        # -----------
        logger.info(f"Training model")
        logger.debug(f"Model: \n{clf}")
        clf.fit(data.X_train, data.y_train)
        score = clf.score(data.X_train, data.y_train)
        logger.info(f"Score on train: {score}")

        # -----------------------
        # Get predictions on test
        # -----------------------
        logger.info("Getting predictions from test data")
        predictions = clf.predict(data.X_test)
        if data.y_test is not None:
            # Lazy import
            from sklearn.metrics import accuracy_score

            test_score = accuracy_score(data.y_test, predictions)
            logger.info(f"Score on test: {test_score}")
