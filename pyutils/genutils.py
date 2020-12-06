"""General utilities
"""
import argparse
import codecs
import glob
import importlib
import json
import logging.config
import os
import shlex
import shutil
import subprocess
import sys
import warnings
from collections import OrderedDict
from logging import NullHandler
from runpy import run_path
from warnings import warn

import pyutils
from pyutils import (CONFIGS_DIRNAME, MODEL_CONFIGS_DIRNAME, MODEL_FNAME_SUFFIX,
                     SKLEARN_MODULES)
from pyutils.default_mlconfigs import __path__ as default_configs_dirpath
default_configs_dirpath = default_configs_dirpath[0]


# Ref.: https://stackoverflow.com/a/26433913/14664104
def warning_on_one_line(message, category, filename, lineno, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)


# TODO: remove?
def get_short_logger_name(name):
    return '.'.join(name.split('.')[-2:])


warnings.formatwarning = warning_on_one_line

logger = logging.getLogger(__name__)
logger.addHandler(NullHandler())

logger_data = logging.getLogger('data')
logger_data.addHandler(NullHandler())

CFG_TYPES = ['main', 'log']


class ConfigBoilerplate:

    # eg. module_file = 'train_models.py'
    def __init__(self, module_file):
        assert module_file in ['explore_data.py', 'train_models.py'], \
            f"Invalid script name: {module_file}"
        # e.g. _module_file = 'train_models'
        self._module_file = os.path.basename(os.path.splitext(module_file)[0])
        # e.g. _package_name = 'titanic'
        self._package_name = os.path.basename(os.getcwd())
        self._package_path = os.getcwd()
        self.module = importlib.import_module(self._module_file)
        self._module_name = self.module.__name__
        self.main_cfg_filepath = get_main_config_filepath()
        self.log_filepath = get_logging_filepath()
        # Get logging cfg dict
        self.log_dict = load_cfg_dict(self.log_filepath, 'log')
        # ============================================
        # Parse command-line arguments and load config
        # ============================================
        if self._module_file == 'train_models':
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
        if self._module_file == 'train_models':
            return self.cfg_dicts
        else:
            return self.cfg_dicts[0]

    # TODO: fix this function
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

    @ staticmethod
    def _override_default_cfg_dict(new_cfg_dict, cfg_type):
        log_msgs = {'cfg': [], 'log': []}
        # Get default cfg dict
        if cfg_type == 'cfg':
            default_cfg_dict = load_cfg_dict(get_main_config_filepath(), cfg_type)
        elif cfg_type == 'log':
            # cfg_type = 'log'
            default_cfg_dict = load_cfg_dict(get_logging_filepath(), cfg_type)
        else:
            raise ValueError(f"Invalid cfg_type: {cfg_type}")
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
        cfg_dict = load_cfg_dict(self.main_cfg_filepath, 'cfg')
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
            list_model_info(show_all=False)
            sys.exit(0)

        if args.list_models:
            list_model_info()
            sys.exit(0)

        # ---------------------
        # No arguments provided
        # ---------------------
        if args.categories is None and args.model_type is None \
                and not args.models:
            # TODO: done too in _parse_cmdl_args_for_explore_script()
            cfg_dict = load_cfg_dict(self.main_cfg_filepath, 'cfg')
            cfg_data['cfg_dicts'].append(cfg_dict)
            cfg_data['cfg_filepaths'].append(self.main_cfg_filepath)
            # TODO: filter warnings
            warn("No arguments provided to the script. Default model "
                 f"'{cfg_dict['model']['model_type']}' from main configuration "
                 "file will be used.", stacklevel=2)
            return cfg_data

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
            args.models, '.py', ['__init__.py'])
        if not model_config_filepaths:
            raise ValueError("No model config files could be retrieved. Check "
                             "the model names or categories provided to the "
                             "script.")
        for cfg_fp in model_config_filepaths:
            # Get config dict
            cfg_dict = load_cfg_dict(cfg_fp, 'cfg')
            # Override default cfg dict with user-defined cfg dict
            log_msgs = self._override_default_cfg_dict(cfg_dict, 'cfg')
            cfg_dict['_log_msgs'] = log_msgs
            cfg_data['cfg_dicts'].append(cfg_dict)
            cfg_data['cfg_filepaths'].append(cfg_fp)
        return cfg_data

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
        # TODO: package name too? instead of program name (e.g. train_models.py)
        parser.add_argument("--version", action='version',
                            version='%(prog)s v{}'.format(pyutils.__version__))
        return parser

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
        # TODO: package name too? instead of program name (e.g. train_models.py)
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
            "-c", "--categories", dest="categories", nargs="+",
            help='''Categories of ML models for which models will be trained.
            These categories correspond to sklearn packages of ML models, e.g. 
            ensemble or linear_model. Use the -l argument to show a complete
            list of all the ML model categories.''')
        parser.add_argument(
            "-m", "--models", dest="models", nargs="+",
            help='''Names of ML models that will be trained. These correspond
            to sklearn classes of ML models, e.g. SVC or AdaBoostClassifier.
            Use the -lm argument to show a complete list of all the supported 
            ML models. Accept model name abbreviations as shown in the list.''')
        parser.add_argument(
            "-t", "--model_type", dest="model_type", choices=['clf', 'reg'],
            default=None,
            help='''The type of model for which models will be trained. `clf`
            is for classifier and `reg` is for regressor.''')
        return parser

    def _setup_log_from_cfg(self):
        # NOTE: if quiet and verbose are both activated, only quiet will have an effect
        # TODO: get first cfg_dict to setup log (same in train_models.py)
        if self.cfg_dicts[0]['verbose'] and not self.cfg_dicts[0]['quiet']:
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
        logger.debug(f"Main config path: {get_main_config_filepath()}")
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
            shutil.copy(fp, dest)


def get_config_dict(cfg_type):
    if cfg_type == 'main':
        cfg_filepath = get_main_config_filepath()
    elif cfg_type == 'log':
        cfg_filepath = get_logging_filepath()
    else:
        raise ValueError(f"Invalid cfg_type: {cfg_type}")
    return load_cfg_dict(cfg_filepath, cfg_type)


def get_configs_dirpath():
    from mlconfigs import __path__
    return __path__[0]


def get_default_configs_dirpath():
    from pyutils.default_configs import __path__
    return __path__[0]


def get_default_scripts_dirpath():
    from pyutils.default_scripts import __path__
    return __path__[0]


def get_default_model_configs_dirpath():
    # Path to the default model_configs directory
    return os.path.join(default_configs_dirpath, MODEL_CONFIGS_DIRNAME)


def get_main_config_filepath():
    return os.path.join(get_configs_dirpath(), 'config.py')


def get_model_configs_dirpath():
    # Path to the model_configs directory in the current working directory
    return os.path.join(os.getcwd(), CONFIGS_DIRNAME, MODEL_CONFIGS_DIRNAME)


def get_logging_filepath():
    return os.path.join(get_configs_dirpath(), 'logging.py')


def remove_ext(filename):
    return os.path.splitext(filename)[0]


# TODO: explain cases
def get_logger_name(module__name__, module___file__, package_name=None):
    if os.path.isabs(module___file__):
        # e.g. initcwd or editcfg
        module_name = os.path.splitext(os.path.basename(module___file__))[0]
        package_path = os.path.dirname(module___file__)
        package_name = os.path.basename(package_path)
        logger_name = "{}.{}".format(
            package_name,
            module_name)
    elif module__name__ == '__main__' or not module__name__.count('.'):
        # e.g. train_models.py or explore_data.py
        if package_name is None:
            package_name = os.path.basename(os.getcwd())
        logger_name = "{}.{}".format(
            package_name,
            os.path.splitext(module___file__)[0])
    elif module__name__.count('.') > 1:
        logger_name = '.'.join(module__name__.split('.')[-2:])
    else:
        # e.g. importing mlutils from train_models.py
        logger_name = module__name__
    return logger_name


# TODO: use file_pattern (regex)
def get_model_config_filepaths(root, categories=None, model_type=None,
                               model_names=None, fname_suffix=MODEL_FNAME_SUFFIX,
                               ignore_fnames=None, lower_model_names=True):
    if categories is None:
        categories = []
    if model_names is None:
        model_names = []
    if ignore_fnames is None:
        ignore_fnames = []
    if model_type:
        model_type = model_type.lower()
        assert model_type in ['classifiers', 'regressors'], \
            f"Invalid model type: {model_type}"
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
            if lower_model_names:
                model_name_found = False
                for name in model_names:
                    if current_model_name.lower() == name.lower():
                        model_name_found = True
                        break
            else:
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


def get_settings(conf, cfg_type):
    if cfg_type == 'log':
        return conf['logging']
    elif cfg_type == 'main':
        _settings = {}
        for opt_name, opt_value in conf.items():
            if opt_name.startswith('__') and opt_name.endswith('__'):
                continue
            elif isinstance(opt_value, type(os)):
                # e.g. import config
                continue
            else:
                _settings.setdefault(opt_name, opt_value)
        return _settings
    else:
        raise ValueError(f"Invalid cfg_type: {cfg_type}")


def is_substring(string, substrings, lower=True):
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


def list_model_info(show_all=True, abbreviations=None, print_msgs=True):
    msgs = ""
    abbr_dict = {}
    default_abbreviations = {
        'CategoricalNB': 'CatNB',
        'ComplementNB': 'ComNB',
        'ExtraTreeClassifier': 'ETC',
        'ExtraTreesClassifier': 'EETC',
        'ExtraTreeRegressor': 'ETR',
        'ExtraTreesRegressr': 'EETR',
        'LinearRegression': 'LiR',
        'LogisticRegression': 'LoR'}
    if abbreviations is None:
        abbreviations = default_abbreviations
    else:
        abbreviations = default_abbreviations.update(abbreviations)
    if show_all:
        title = "***List of model categories and the associated ML models***"
    else:
        title = "***List of model categories***"
    msgs += title
    acronyms = []
    for i, module in enumerate(SKLEARN_MODULES, start=1):
        spaces = '  ' if i < 10 else ' '
        # e.g. (1)  ensemble
        msgs += f"\n({i}){spaces}{module}"
        # Path to the model configs folder in the working directory
        module_dirpath = os.path.join(get_model_configs_dirpath(), module)
        if module  == 'ensemble':
            import ipdb
            ipdb.set_trace()
        if not os.path.exists(module_dirpath):
            module_dirpath = os.path.join(get_default_model_configs_dirpath(),
                                          module)
        if show_all:
            # For each sklearn module, print its associated model names
            # Get all python files (model config files) under the sklearn module directory
            for path, subdirs, files in os.walk(module_dirpath):
                for i, fname in enumerate(files):
                    # Retrieve the model name from the python config filename
                    # e.g. DummyClassifier_config.py -> DummyClassifier
                    model_name = fname.split(MODEL_FNAME_SUFFIX)
                    if os.path.splitext(fname)[1] != '.py':
                        # Ignore non-python files, e.g. .DS_Store
                        continue
                    elif len(model_name) == 1:
                        # Ignore fname since it doesn't have a valid suffix
                        continue
                    else:
                        model_name = model_name[0]
                        if i == 0:
                            # i.e. classifiers or regressors
                            model_type = os.path.basename(path)
                            msgs += f"\n\t* {model_type}"

                        def get_acronym(compound_word):
                            return ''.join([l for l in compound_word if not l.islower()])

                        if abbreviations.get(model_name):
                            short_name = abbreviations.get(model_name)
                        else:
                            acronym = get_acronym(model_name)
                            i = 1
                            while acronym in acronyms:
                                acronym = f"{acronym}{i}"
                                i += 1
                            acronyms.append(acronym)
                            short_name = acronym
                        msgs += f"\n\t    - {model_name} [{short_name}]"
                        abbr_dict.setdefault(short_name.lower(), model_name)
    if show_all:
        msgs += "\n\nNotes:\n- Beside each number in parentheses, it is the " \
                "model category\n- Between brackets, it is the model name " \
                "abbreviation\n"
    if print_msgs:
        print(msgs)
    return abbr_dict


def load_cfg_dict(cfg_filepath, cfg_type):
    assert cfg_type in CFG_TYPES, f"Invalid cfg_type: {cfg_type}"
    _, file_ext = os.path.splitext(cfg_filepath)
    try:
        if file_ext == '.py':
            cfg_dict = run_path(cfg_filepath)
            cfg_dict = get_settings(cfg_dict, cfg_type)
        elif file_ext == '.json':
            cfg_dict = load_json(cfg_filepath)
        else:
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


def run_cmd(cmd):
    """Run a shell command with arguments.

    The shell command is given as a string but the function will split it in
    order to get a list having the name of the command and its arguments as
    items.

    Parameters
    ----------
    cmd : str
        Command to be executed, e.g. ::

            open -a TextEdit text.txt

    Returns
    -------
    retcode: int
        Returns code which is 0 if the command was successfully completed.
        Otherwise, the return code is non-zero.

    Raises
    ------
    FileNotFoundError
        Raised if the command ``cmd`` is not recognized, e.g.
        ``$ TextEdit {filepath}`` since `TextEdit` is not an executable.

    """
    try:
        if sys.version_info.major == 3 and sys.version_info.minor <= 6:
            # TODO: PIPE not working as arguments and capture_output new in
            # Python 3.7
            # Ref.: https://stackoverflow.com/a/53209196
            #       https://bit.ly/3lvdGlG
            result = subprocess.run(shlex.split(cmd))
        else:
            result = subprocess.run(shlex.split(cmd), capture_output=True)
    except FileNotFoundError:
        raise
    else:
        return result


def process_model_names(model_names):
    processed_names = []
    for model_name in model_names:
        abbr_dict = list_model_info(print_msgs=False)
        model_name = model_name.lower()
        if abbr_dict.get(model_name):
            processed_names.append(abbr_dict.get(model_name))
        else:
            processed_names.append(model_name)
    return processed_names



def set_logging_level(log_dict, level='DEBUG'):
    keys = ['handlers', 'loggers']
    for k in keys:
        for name, val in log_dict[k].items():
            val['level'] = level


def setup_log(quiet=False, verbose=False):
    package_path = os.getcwd()
    log_filepath = get_logging_filepath()
    # Get logging cfg dict
    log_dict = load_cfg_dict(log_filepath, cfg_type='log')
    # NOTE: if quiet and verbose are both activated, only quiet will have an effect
    # TODO: get first cfg_dict to setup log (same in train_models.py)
    if not quiet:
        # Load logging config dict
        if verbose:
            set_logging_level(log_dict)
        logging.config.dictConfig(log_dict)
    # =============
    # Start logging
    # =============
    logger.info("Running {} v{}".format(pyutils.__name__, pyutils.__version__))
    logger.info("Verbose option {}".format(
        "enabled" if verbose else "disabled"))
    logger.debug("Working directory: {}".format(package_path))
    logger.debug(f"Main config path: {get_main_config_filepath()}")
    logger.debug(f"Logging path: {log_filepath}")
