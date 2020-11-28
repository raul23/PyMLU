"""Machine learning utilities
"""
import importlib
import logging.config
from logging import NullHandler

# import ipdb
# import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from datasets.my_utils import genutils as ge

pandas = None

logger = logging.getLogger(ge.get_short_logger_name(__name__))
logger.addHandler(NullHandler())

_SKLEARN_MODULES = ['sklearn.dummy', 'sklearn.gaussian_process',
                    'sklearn.linear_model', 'sklearn.naive_bayes',
                    'sklearn.neighbors', 'sklearn.neural_network'
                    'sklearn.semi_supervised', 'sklearn.semi_supervised',
                    'sklearn.svm', 'sklearn.tree', 'sklearn.calibration',
                    'sklearn.ensemble', 'sklearn.multiclass',
                    'sklearn.multioutput']


class Datasets:

    def __init__(self, train_filepath, test_filepath, y_target, features=None,
                 get_dummies=False, *args, **kwargs):
        # ------------------
        # Parameters parsing
        # ------------------
        global pandas
        logger.info("Importing pandas...")
        import pandas
        self.train_filepath = train_filepath
        self.test_filepath = test_filepath
        self.y_target = y_target
        if features:
            logger.debug(f"Using only these features: {features}")
            self.features = features
        else:
            logger.debug("Using all features")
            self.features = None
        self.get_dummies = get_dummies
        # ------------
        # Data loading
        # ------------
        # Load train data
        logger.info("Loading training data")
        self.train_data = pandas.read_csv(train_filepath)
        if not self.features:
            self.features = self.train_data.columns.to_list()
            # Remove target from features
            self.features.remove(self.y_target)
            logger.debug(f"Using all {len(self.features)} features")
        self.y = self.train_data[y_target]
        # Load test data
        logger.info("Loading test data")
        self.test_data = pandas.read_csv(test_filepath)
        # ------------------
        # Data preprocessing
        # ------------------
        # Select only the required features
        # TODO: remove target from train X
        self.X = self.train_data[self.features]
        self.X_test = self.test_data[self.features]
        if self.get_dummies:
            self._get_dummies()

    # One-hot encode the data
    def _get_dummies(self):
        # Replace missing values on train and test
        logger.info("One-hot encoding the data")
        self.X = pandas.get_dummies(self.X)
        self.X_test = pandas.get_dummies(self.X_test)


def get_model(model_type, model_params, scale_input=False):
    logger.debug(f"Get model: {model_type}")
    model_type_split = model_type.split('.')
    assert len(model_type_split), \
        "There should be three components to the model type. Only " \
        f"{len(model_type)} provided: {model_type}"
    sklearn_module = '.'.join(model_type_split[:2])
    model_name = model_type_split[-1]
    if sklearn_module in _SKLEARN_MODULES:
        logger.info(f"Importing {model_type}...")
        module = importlib.import_module(sklearn_module)
    else:
        raise TypeError(f"The model type is invalid: {model_type}")
    if model_name == 'HistGradientBoostingClassifier':
        # Note: this estimator is still experimental for now: To use it, you need to
        #       explicitly import enable_hist_gradient_boosting
        # Ref.: https://bit.ly/3ldqWKp
        exp_module_name = 'sklearn.experimental.enable_hist_gradient_boosting'
        logger.info(f"Importing experimental module: {exp_module_name}")
        importlib.import_module(exp_module_name)
    model_class = getattr(module, model_name)
    logger.debug(f"Model imported: {model_class}")
    base_estimator_cfg = model_params.get('base_estimator')
    if base_estimator_cfg is None:
        base_estimator_cfg = model_params.get('estimator')
    estimators_cfg = model_params.get('estimators')
    if base_estimator_cfg:
        # e.g. AdaBoostClassifier
        base_model = get_model(**base_estimator_cfg)
        model = model_class(base_model, **model_params)
    elif estimators_cfg:
        # e.g. StackingClassifier
        base_estimators = []
        for base_estimator_cfg in estimators_cfg:
            estimator_name = ''.join(c for c in base_estimator_cfg['model_type']
                                     if c.isupper())
            base_estimators.append((estimator_name, get_model(**base_estimator_cfg)))
        final_estimator_cfg = model_params.get('final_estimator')
        final_estimator = None
        if final_estimator_cfg:
            final_estimator = get_model(**final_estimator_cfg)
        else:
            # TODO: log (final estimator is None)
            pass
        model = model_class(estimators=base_estimators,
                            final_estimator=final_estimator)
    else:
        # Only the ensemble method, e.g. RandomForestClassifier
        model = model_class(**model_params)
    if scale_input:
        logger.info("Input will be scaled")
        """
        logger.info("Importing sklearn.pipeline.make_pipeline")
        from sklearn.pipeline import make_pipeline
        """
        return make_pipeline(StandardScaler(), model)
    else:
        return model
