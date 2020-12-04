"""General utilities
"""
import argparse
import codecs
import glob
import importlib
import json
import logging.config
import os
import sys
from shutil import copy
from collections import OrderedDict
from logging import NullHandler
from runpy import run_path

import pyutils
from pyutils import (CONFIGS_DIRNAME, MODEL_CONFIGS_DIRNAME, MODEL_FNAME_SUFFIX,
                     SKLEARN_MODULES)
from pyutils.default_configs import __path__ as default_configs_dirpath
default_configs_dirpath = default_configs_dirpath[0]


def get_short_logger_name(name):
    return '.'.join(name.split('.')[-2:])


logger = logging.getLogger(get_short_logger_name(__name__))
logger.addHandler(NullHandler())

logger_data = logging.getLogger('data')
logger_data.addHandler(NullHandler())


class ConfigBoilerplate:

    # eg. module_file = 'train_model.py'
    def __init__(self, module_file):
        assert module_file in ['explore_data.py', 'train_model.py'], \
            f"Invalid script name: {module_file}"
        # e.g. _module_file = 'train_model'
        self._module_file = os.path.basename(os.path.splitext(module_file)[0])
        # e.g. _package_name = 'titanic'
        self._package_name = os.path.basename(os.getcwd())
        self._package_path = os.getcwd()
        self.module = importlib.import_module(self._module_file)
        self._module_name = self.module.__name__
        self.main_cfg_filepath = get_main_cfg_filepath()
        self.log_filepath = get_logging_filepath()
        # Get logging cfg dict
        self.log_dict = load_cfg_dict(self.log_filepath, is_logging=True)
        # ============================================
        # Parse command-line arguments and load config
        # ============================================
        if self._module_file == 'train_model':
            retval = self._parse_cmdl_args_for_train_script()
        else:
            retval = self._parse_cmdl_args_for_explore_script()
        self.cfg_filepaths = retval['cfg_filepaths']
        self.cfg_dicts = retval['cfg_dicts']
        # ==============================
        # Logging setup from config file
        # ==============================
        self._setup_log_from_cfg()

    def get_cfg_dict(self):
        if self._module_file == 'train_model':
            return self.cfg_dicts
        else:
            return self.cfg_dicts[0]

    def _log_overridden_cfgs(self):
        cfg_types_map = {'cfg': 'config dict', 'log': 'logging dict'}
        for cfg_type, cfgs in self._overridden_cfgs.items():
            if cfgs:
                logger.info(f"{len(cfgs)} option(s) overridden in "
                            f"{cfg_types_map[cfg_type]}")
                log_msg = f"# Overridden options in {cfg_types_map[cfg_type]}"
                equals_line = "# " + (len(log_msg) - 2) * "="
                logger_data.debug(f"{equals_line}\n{log_msg}\n{equals_line}")
                for cfg in cfgs:
                    logger_data.debug(cfg)
                logger_data.debug("")

    # TODO: cfg_type = {'cfg', 'log'}
    def _override_default_cfg_dict(self, new_cfg_dict, cfg_type):
        assert cfg_type in ['cfg', 'log'], f"Invalid cfg_type: {cfg_type}"
        log_msgs = {'cfg': [], 'log': []}
        # Get default cfg dict
        if cfg_type == 'cfg':
            default_cfg_dict = load_cfg_dict(get_main_cfg_filepath(),
                                             is_logging=False)
        else:
            # cfg_type = 'log'
            default_cfg_dict = load_cfg_dict(get_logging_filepath(),
                                             is_logging=True)
        for k, v in default_cfg_dict.items():
            if new_cfg_dict.get(k) is None:
                new_cfg_dict[k] = v
            else:
                if new_cfg_dict[k] != v:
                    if len(f"{v}") > 65 or len(f"{new_cfg_dict[k]}") > 65:
                        msg = f"** {k} **:\n{v}\n| -> {new_cfg_dict[k]}"
                    else:
                        msg = f"** {k} **: {v} -> {new_cfg_dict[k]}"
                    log_msgs[cfg_type].append(msg)
                    v = new_cfg_dict[k]
        return log_msgs

    def _parse_cmdl_args_for_explore_script(self):
        cfg_data = {'cfg_filepaths': [], 'cfg_dicts': []}
        parser = self._setup_argparser_for_explore_script()
        args = parser.parse_args()
        # Get config dict
        cfg_dict = load_cfg_dict(self.main_cfg_filepath, is_logging=False)
        cfg_data['cfg_dicts'].append(cfg_dict)
        cfg_data['cfg_filepaths'].append(self.main_cfg_filepath)
        return cfg_data

    def _parse_cmdl_args_for_train_script(self):
        cfg_data = {'cfg_filepaths': [], 'cfg_dicts': []}
        parser = self._setup_argparser_for_train_script()
        args = parser.parse_args()

        # --------------------------------------------------------------------
        # -l and -lm : list model categories and/or their associated ML models
        # --------------------------------------------------------------------
        if args.list_categories:
            list_model_categories_and_names(show_all=False)
            sys.exit(0)

        if args.list_models:
            list_model_categories_and_names()
            sys.exit(0)

        # --------------------------------------------------
        # -c -t -m: categories, types and names of ML models
        # --------------------------------------------------
        if args.categories:
            for cat in args.categories:
                if cat not in SKLEARN_MODULES:
                    raise ValueError(f"Invalid model category: {cat}. Run the "
                                     "script with the -l argument to get the "
                                     "complete list of all the supported "
                                     "categories")
        # if (isinstance(args.categories, list) and not args.categories) or args.categories:
        if args.categories == [] or args.categories:
            assert args.model_type, \
                "the following arguments are required: -t/--model_type"
        # if isinstance(args.categories, list) and not args.categories:
        if args.categories == [] or \
                (args.categories is None and args.model_type):
            logger.info("**All model categories selected**\n")
            args.categories = SKLEARN_MODULES
        if args.model_type:
            args.model_type = 'classifiers' if args.model_type == 'clf' else 'regressors'
        args.models = [] if args.models is None else args.models
        # Path to the model_configs directory in the current working directory
        model_configs_dirpath = os.path.join(os.getcwd(), CONFIGS_DIRNAME,
                                             MODEL_CONFIGS_DIRNAME)
        model_config_filepaths = get_model_config_filepaths(
            model_configs_dirpath, args.categories, args.model_type,
            args.models, '.py')
        if not model_config_filepaths:
            raise ValueError("No model config files could be retrieved. Check "
                             "the model names or categories provided to the "
                             "script.")
        for cfg_fp in model_config_filepaths:
            # Get config dict
            cfg_dict = load_cfg_dict(cfg_fp, is_logging=False)
            # Override default cfg dict with user-defined cfg dict
            log_msgs = self._override_default_cfg_dict(cfg_dict, 'cfg')
            cfg_dict['_log_msgs'] = log_msgs
            cfg_data['cfg_dicts'].append(cfg_dict)
            cfg_data['cfg_filepaths'].append(cfg_fp)
        return cfg_data

    @staticmethod
    def _setup_argparser_for_train_script():
        """Setup the argument parser for the command-line script.

        TODO

        Returns
        -------
        parser : argparse.ArgumentParser
            Argument parser.

        """
        # Setup the parser
        parser = argparse.ArgumentParser(
            # usage="%(prog)s [OPTIONS]",
            # prog=os.path.basename(__file__),
            description='''\
    TODO\n''',
            # formatter_class=argparse.RawDescriptionHelpFormatter)
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # TODO: package name too? instead of program name (e.g. train_model.py)
        parser.add_argument("--version", action='version',
                            version='%(prog)s v{}'.format(pyutils.__version__))
        parser.add_argument(
            "-l", "--list-categories", dest="list_categories", action='store_true',
            help='''Show a list of all the supported ML model categories. Then 
            the program exits.''')
        parser.add_argument(
            "-lm", "--list-models", dest="list_models", action='store_true',
            help='''Show a list of all the supported ML models. Then the 
            program exits.''')
        parser.add_argument(
            "-c", "--categories", dest="categories", nargs="*",
            help='''Categories of ML models for which models will be trained.
            These categories correspond to sklearn packages of ML models, e.g. 
            ensemble or linear_model. Use the -l argument to show a complete 
            list of all the ML model categories.''')
        parser.add_argument(
            "-m", "--models", dest="models", nargs="*",
            help='''Names of ML models that will be trained. These correspond
            to sklearn classes of ML models, e.g. SVC or AdaBoostClassifier.
            Use the -lm argument to show a complete list of all the supported 
            ML models.''')
        parser.add_argument(
            "-t", "--model_type", dest="model_type", choices=['clf', 'reg'],
            default=None,
            help='''The type of model for which models will be trained. `clf`
            is for classifier and `reg` is for regressor.''')
        return parser

    @staticmethod
    def _setup_argparser_for_explore_script():
        """Setup the argument parser for the command-line script.

        TODO

        Returns
        -------
        parser : argparse.ArgumentParser
            Argument parser.

        """
        # Setup the parser
        parser = argparse.ArgumentParser(
            # usage="%(prog)s [OPTIONS]",
            # prog=os.path.basename(__file__),
            description='''\
        TODO\n''',
            # formatter_class=argparse.RawDescriptionHelpFormatter)
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # TODO: package name too? instead of program name (e.g. train_model.py)
        parser.add_argument("--version", action='version',
                            version='%(prog)s v{}'.format(pyutils.__version__))
        return parser

    def _setup_log_from_cfg(self):
        module_logger = logging.getLogger(
            get_logger_name(self._package_name,
                            self._module_name,
                            self._module_file))
        # NOTE: if quiet and verbose are both activated, only quiet will have an effect
        # TODO: get first cfg_dict to setup log (same in train_model.py)
        if self.cfg_dicts[0]['quiet']:
            # TODO: disable logging completely? even error messages?
            module_logger.disabled = True
        else:
            # Load logging config dict
            if self.cfg_dicts[0]['verbose']:
                set_logging_level(self.log_dict)
            logging.config.dictConfig(self.log_dict)
        # =============
        # Start logging
        # =============
        logger.info("Running {} v{}".format(pyutils.__name__, pyutils.__version__))
        # logger.info("Using the dataset: {}".format(self._package_name))
        logger.info("Verbose option {}".format(
            "enabled" if self.cfg_dicts[0]['verbose'] else "disabled"))
        logger.debug("Working directory: {}".format(self._package_path))
        logger.debug(f"Main config path: {get_main_cfg_filepath()}")
        logger.debug(f"Logging path: {self.log_filepath}")


def copy_files(src_dirpath, dest_dirpath, width=(1,1), file_pattern='*.*',
               overwrite=False):
    for fp in glob.glob(os.path.join(src_dirpath, file_pattern)):
        fname = os.path.basename(fp)
        dest = os.path.join(dest_dirpath, fname)
        if os.path.exists(dest) and not overwrite:
            print(f"{'File ' + fname + ' exists':{width[0]}s}: {dest}")
            print(f"Skipping it!")
            continue
        else:
            # TODO: copy2?
            print(f"Copying {fname:{width[1]}s} to {dest}")
            copy(fp, dest)


def get_cfgs_dirpath():
    from configs import __path__ as configs_path
    return configs_path[0]


def get_main_cfg_filepath():
    return os.path.join(get_cfgs_dirpath(), 'config.py')


def get_logging_filepath():
    return os.path.join(get_cfgs_dirpath(), 'logging.py')


# TODO: module_file must be the filename (not whole filepath)
def get_logger_name(module_name, module_file, package_name=None):
    if package_name is None:
        package_name = os.path.basename(os.getcwd())
    if module_name == '__main__' or not module_name.count('.'):
        logger_name = "{}.{}".format(
            package_name,
            os.path.splitext(module_file)[0])
    elif module_name.count('.') > 1:
        logger_name = '.'.join(module_name.split('.')[-2:])
    else:
        logger_name = module_name
    return logger_name


# TODO: remove?
def get_model_config_path(root, model_name):
    filepath = None
    for path, subdirs, files in os.walk(root):
        for name in files:
            if model_name.lower() in name.lower():
                filepath = os.path.join(path, name)
                import ipdb
                ipdb.set_trace()
                break
    return filepath


# TODO: use file_pattern (regex)
def get_model_config_filepaths(root, categories=None, model_type=None,
                               model_names=None, fname_suffix=MODEL_FNAME_SUFFIX,
                               ignore_fnames=['__init__.py'], lower=True):
    if model_type:
        model_type = model_type.lower()
        assert model_type in ['classifiers', 'regressors'], \
            f"Invalid model type: {model_type}"
    if categories is None:
        categories = []
    filepaths = []
    for path, subdirs, files in os.walk(root):
        for fname in files:
            if fname in ignore_fnames or not fname.endswith(fname_suffix):
                # fname should be ignore or doesn't have the correct suffix;
                # next fname
                continue
            # else: fname has the correct suffix
            # Search the fname for one of the model_names
            # TODO: add it in a function (used somewhere else too)
            current_model_name = os.path.basename(fname).split(MODEL_FNAME_SUFFIX)[0]
            current_model_type = os.path.basename(path)
            current_model_category = os.path.basename(os.path.dirname(path))
            model_name_found = current_model_name in model_names
            if model_name_found and model_type and model_type != current_model_type:
                raise ValueError(f"Trying to train a model ({current_model_name}) "
                                 "that is different from the specified model_type "
                                 f"({model_type})")
            if model_name_found or (current_model_type==model_type and
                                    current_model_category in categories) or \
                (model_type is None and current_model_category in categories):
                # Add the fname since it is a valid one
                filepaths.append(os.path.join(path, fname))
            else:
                # fname not part of a valid dirname (category) or not correct
                # model type or model name not found in the fname; next fname
                continue
    return filepaths


def get_settings(conf, is_logging=False):
    _settings = {}
    if is_logging:
        return conf['logging']
    for opt_name, opt_value in conf.items():
        if opt_name.startswith('__') and opt_name.endswith('__'):
            continue
        elif isinstance(opt_value, type(os)):
            # e.g. import config
            continue
        else:
            _settings.setdefault(opt_name, opt_value)
    return _settings


def is_substring_in_string(string, substrings, lower=True):
    # Search the string for one of the substrings
    substring_found = False
    for subs in substrings:
        string_copy = string
        if lower:
            subs = subs.lower()
            string_copy = string_copy.lower()
        # else: don't lowercase the string
        if string_copy.count(subs):
            substring_found = True
            # Valid string: contains the substring
            break
        # else: string doesn't have the correct substring; next substring
    return substring_found


def list_model_categories_and_names(show_all=True):
    if show_all:
        title = "***List of model categories [C] and the associated ML models [M]***"
        module_msg_ending = "[C]:"
    else:
        title = "***List of model categories***"
        module_msg_ending = ""
    print(title)
    for i, module in enumerate(SKLEARN_MODULES, start=1):
        spaces = '  ' if i < 10 else ' '
        # e.g. (1)  ensemble
        print(f"({i}){spaces}{module} {module_msg_ending}")
        # Path to the sklearn module directory
        module_dirpath = os.path.join(default_configs_dirpath,
                                      MODEL_CONFIGS_DIRNAME, module)
        if show_all:
            # For each sklearn module, print its associated model names
            # Get all python files (model config files) under the sklearn module directory
            for path, subdirs, files in os.walk(module_dirpath):
                if files:
                    model_type = os.path.basename(path)
                    print(f"\t* {model_type}")
                for fname in files:
                    # Retrieve the model name from the python config filename
                    # e.g. DummyClassifier_config.py -> DummyClassifier
                    model_name = os.path.basename(fname).split(MODEL_FNAME_SUFFIX)[0]
                    print(f"\t    - {model_name} [M]")
    if show_all:
        print("\nLegend:\n[C]: model category\n[M]: model name\n")


def load_cfg_dict(cfg_filepath, is_logging):
    _, file_ext = os.path.splitext(cfg_filepath)
    try:
        if file_ext == '.py':
            cfg_dict = run_path(cfg_filepath)
            cfg_dict = get_settings(cfg_dict, is_logging)
        elif file_ext == '.json':
            cfg_dict = load_json(cfg_filepath)
        else:
            # TODO: log error message
            raise FileNotFoundError(
                f"[Errno 2] No such file or directory: "
                f"{cfg_filepath}")
    except FileNotFoundError as e:
        raise e
    else:
        return cfg_dict


def load_json(filepath, encoding='utf8'):
    """Load JSON data from a file on disk.

    If using Python version betwee 3.0 and 3.6 (inclusive), the data is
    returned as :obj:`collections.OrderedDict`. Otherwise, the data is
    returned as :obj:`dict`.

    Parameters
    ----------
    filepath : str
        Path to the JSON file which will be read.
    encoding : str, optional
        Encoding to be used for opening the JSON file in read mode (the default
        value is '*utf8*').

    Returns
    -------
    data : dict or collections.OrderedDict
        Data loaded from the JSON file.

    Raises
    ------
    OSError
        Raised if any I/O related error occurs while reading the file, e.g. the
        file doesn't exist.

    References
    ----------
    `Are dictionaries ordered in Python 3.6+? (stackoverflow)`_

    """
    try:
        with codecs.open(filepath, 'r', encoding) as f:
            if sys.version_info.major == 3 and sys.version_info.minor <= 6:
                data = json.load(f, object_pairs_hook=OrderedDict)
            else:
                data = json.load(f)
    except OSError:
        raise
    else:
        return data


def mkdir(path, widths=(1, 1)):
    dirname = os.path.basename(path)
    if os.path.exists(path):
        print(f"'{dirname}' folder {'exists':{widths[0]}s}: {path}")
        # print(f"Skipping it!")
    else:
        print(f"Creating the '{dirname}' {'folder':{widths[1]}s}: {path}")
        os.mkdir(path)


def set_logging_level(log_dict, level='DEBUG'):
    keys = ['handlers', 'loggers']
    for k in keys:
        for name, val in log_dict[k].items():
            val['level'] = level
