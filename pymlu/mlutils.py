"""Machine learning utilities
"""
import importlib
import logging
from logging import NullHandler

from pymlu import SKLEARN_MODULES
from pymlu.default_mlmodules.train_models import train
from pymlu.genutils import get_config_from_locals, get_configs

numpy = None
pandas = None

logger = logging.getLogger(__name__)
logger.addHandler(NullHandler())

logger_data = logging.getLogger('data')
logger_data.addHandler(NullHandler())


class Dataset:

    def __init__(self, builtin_dataset=None, custom_dataset=None,
                 use_custom_data=False, features=None, get_dummies=False,
                 random_seed=0, config=None, *args, **kwargs):
        cfg = get_config_from_locals(config, locals(), ignored_keys=['config'])
        global numpy, pandas
        logger.debug("Importing numpy and pandas...")
        # Lazy import
        import numpy
        import pandas
        # ------------------
        # Parameters parsing
        # ------------------
        assert cfg.builtin_dataset or cfg.custom_dataset, \
            "No dataset specified. You need to specify a builtin or custom " \
            "dataset."
        self.random_seed = cfg.random_seed
        if cfg.use_custom_data:
            assert cfg.custom_dataset, \
                "use_custom_data=True but no custom dataset specified"
            self.custom_dataset = cfg.custom_dataset
            self.builtin_dataset = None
            assert cfg.custom_dataset.train_filepath, \
                "Train filepath is missing"
            assert cfg.custom_dataset.test_filepath, \
                "Test filepath is missing"
            logger.debug(f"y_target = {self.custom_dataset.y_target}")
        else:
            assert cfg.builtin_dataset, "No builtin custom dataset specified"
            self.builtin_dataset = cfg.builtin_dataset
            self.custom_dataset = None
        if cfg.features:
            logger.debug(f"Using only these features: {cfg.features}")
            self.features = cfg.features
        else:
            logger.debug("Using all features")
            self.features = None
        self.get_dummies = cfg.get_dummies
        # ------------
        # Data loading
        # ------------
        # Init data variables
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.data_types = ['train', 'valid', 'test']
        if cfg.use_custom_data:
            self._process_custom_dataset()
        else:
            self._process_builtin_dataset()
        self._print_data_info()
        # ------------------
        # Data preprocessing
        # ------------------
        # One-hot encode the data
        if self.get_dummies:
            self._get_dummies()

    def _print_data_info(self):
        for data_type in self.data_types:
            X_data, y_data = self.get_data(data_type)
            if X_data is not None:
                logger.debug(f"X_{data_type} shape: {X_data.shape}")
            if y_data is not None:
                logger.debug(f"y_{data_type} shape: {y_data.shape}")

    def get_data(self, data_type):
        try:
            X_data = getattr(self, 'X_' + data_type)
            y_data = getattr(self, 'y_' + data_type)
        except AttributeError:
            X_data = None
            y_data = None
        return X_data, y_data

    @staticmethod
    def shuffle_dataset(X, y, random_seed=1):
        n_sample = len(X)
        numpy.random.seed(random_seed)
        order = numpy.random.permutation(n_sample)
        X = X.loc[order]
        y = y.loc[order]
        return X, y

    @staticmethod
    def select_features(features, data):
        if features:
            if isinstance(features[0], str):
                data = data[features]
            else:
                data = data.iloc[:, features]
        return data

    @staticmethod
    def split_data(X, y, data_prop=(0.9, 0.1)):
        assert sum(data_prop) == 1, \
            f"The sum of the data proportions is not 1.0: {data_prop}"
        n_sample = len(X)
        X_train = X[:int(data_prop[0] * n_sample)]
        y_train = y[:int(data_prop[0] * n_sample)]
        X_test = X[int(data_prop[0] * n_sample):]
        y_test = y[int(data_prop[0] * n_sample):]
        return X_train, y_train, X_test, y_test

    # One-hot encode the data
    def _get_dummies(self):
        # Replace missing values on train and test
        logger.debug("One-hot encoding the data")
        self.X_train = pandas.get_dummies(self.X_train)
        self.X_test = pandas.get_dummies(self.X_test)

    def _process_builtin_dataset(self):
        # Lazy import
        from sklearn import datasets
        # ------------
        # Load dataset
        # ------------
        # Load train data
        logger.info(f"Loading dataset {self.builtin_dataset.name}...")
        if self.builtin_dataset.name == 'iris':
            iris = datasets.load_iris()
            self.builtin_dataset.update(iris)
            X = pandas.DataFrame(self.builtin_dataset.pop('data'))
            X.columns = self.builtin_dataset.feature_names
            y = pandas.Series(self.builtin_dataset.pop('target'))
            y.name = 'iris_species'
        else:
            raise ValueError("Dataset not supported: "
                             f"{self.builtin_dataset.name}")
        # ------------------
        # Features selection
        # ------------------
        # Select only the required features
        if self.features:
            logger.debug(f"Using {len(self.features)} features")
            X = self.select_features(self.features, X)
        # --------------
        # Randomize data
        # --------------
        if self.builtin_dataset.shuffle_data:
            logger.debug("Shuffling dataset")
            X, y = self.shuffle_dataset(X, y, self.random_seed)
        # -----------
        # Data splits
        # -----------
        self.X_train, self.y_train, self.X_test, self.y_test = self.split_data(
            X, y, self.builtin_dataset.data_prop)

    def _process_custom_dataset(self):
        # ------------
        # Load dataset
        # ------------
        # Load train data
        logger.debug("Loading training data...")
        logger.info(f"Train filepath: {self.custom_dataset.train_filepath}")
        train_data = pandas.read_csv(self.custom_dataset.train_filepath)
        if not self.features:
            self.features = train_data.columns.to_list()
        # Remove target from features
        if self.custom_dataset.y_target in self.features:
            logger.info("Removing the y_target "
                        f"({self.custom_dataset.y_target}) from the features")
            self.features.remove(self.custom_dataset.y_target)
        X = train_data
        self.y_train = train_data[self.custom_dataset.y_target]
        # Load test data
        logger.debug("Loading test data...")
        logger.info(f"Test filepath: {self.custom_dataset.test_filepath}")
        X_test = pandas.read_csv(self.custom_dataset.test_filepath)
        # ------------------
        # Features selection
        # ------------------
        # Select only the required features
        logger.debug(f"Using {len(self.features)} features")
        if self.features:
            self.X_train = self.select_features(self.features, X)
            self.X_test = self.select_features(self.features, X_test)
        else:
            self.X_train = X
            self.X_test = X_test


def get_model(model_config=None, model_name=None, model_params=None,
              scale_input=False):
    # TODO: eventually check verbose and quiet (need access to log_dict)
    cfg = get_config_from_locals(model_config, locals(),
                                 ignored_keys=['model_config'])
    if not model_config:
        if not cfg.model_name:
            raise ValueError("model_name missing")
        if not cfg.model_params:
            raise ValueError("model_params missing")
    logger.debug(f"Get model: {cfg.model_name}")
    model_name_split = cfg.model_name.split('.')
    assert len(model_name_split), \
        "There should be three components to the model name. Only " \
        f"{len(cfg.model_name)} provided: {cfg.model_name}"
    sklearn_module = '.'.join(model_name_split[:2])
    module_name = model_name_split[1]
    model_classname = model_name_split[2]
    if module_name in SKLEARN_MODULES:
        logger.debug(f"Importing {cfg.model_name}...")
        module = importlib.import_module(sklearn_module)
    else:
        raise TypeError(f"The model name is invalid: {cfg.model_name}")
    if model_classname == 'HistGradientBoostingClassifier':
        # Note: this estimator is still experimental for now: To use it, you need to
        #       explicitly import enable_hist_gradient_boosting
        # Ref.: https://bit.ly/3ldqWKp
        exp_module_name = 'sklearn.experimental.enable_hist_gradient_boosting'
        logger.debug(f"Importing experimental module: {exp_module_name}")
        importlib.import_module(exp_module_name)
    model_class = getattr(module, model_classname)
    logger.debug(f"Model imported: {model_class}")
    # Can either be base_estimator or estimator (equivalent)
    base_estimator_cfg = cfg.model_params.get('base_estimator')
    if base_estimator_cfg is None:
        base_estimator_cfg = cfg.model_params.get('estimator')
    estimators_cfg = cfg.model_params.get('estimators')
    if base_estimator_cfg:
        # e.g. AdaBoostClassifier
        base_model = get_model(**base_estimator_cfg)
        model = model_class(base_model, **cfg.model_params)
    elif estimators_cfg:
        # e.g. StackingClassifier
        base_estimators = []
        for base_estimator_cfg in estimators_cfg:
            estimator_name = ''.join(c for c in base_estimator_cfg['model_type']
                                     if c.isupper())
            base_estimators.append((estimator_name, get_model(**base_estimator_cfg)))
        final_estimator_cfg = cfg.model_params.get('final_estimator')
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
        model = model_class(**cfg.model_params)
    if cfg.scale_input:
        # Lazy import
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        """
        logger.info("Importing sklearn.pipeline.make_pipeline")
        """
        logger.info("Input will be scaled")
        return make_pipeline(StandardScaler(), model)
    else:
        return model


# TODO: from example.py
def train_models(main_config_dict=None, model_configs=None, quiet=False,
                 verbose=False, logging_level=None, logging_formatter=None):
    train(get_configs(**locals()))
