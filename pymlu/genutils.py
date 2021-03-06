"""General utilities
"""
import codecs
import copy
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
from types import SimpleNamespace
from warnings import warn

import pymlu
from pymlu import (CONFIGS_DIRNAME, MODEL_CONFIGS_DIRNAME,
                   MODEL_FNAME_SUFFIX, MODULES_DIRNAME, SKLEARN_MODULES)


# TODO: move it at the bottom
# Ref.: https://stackoverflow.com/a/26433913/14664104
def warning_on_one_line(message, category, filename, lineno, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)


warnings.formatwarning = warning_on_one_line

logger = logging.getLogger(__name__)
logger.addHandler(NullHandler())

logger_data = logging.getLogger('data')
logger_data.addHandler(NullHandler())

CFG_TYPES = ['main', 'log']
DEFAULT_ABBREVIATIONS = {
    'CategoricalNB': 'CatNB',
    'ComplementNB': 'ComNB',
    'ExtraTreeClassifier': 'ETC',
    'ExtraTreesClassifier': 'EETC',
    'ExtraTreeRegressor': 'ETR',
    'ExtraTreesRegressr': 'EETR',
    'LinearRegression': 'LiR',
    'LogisticRegression': 'LoR'}
MODULE_NAMES = ['explore_data', 'train_models']


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
        logger.info("Running {} v{}".format(pymlu.__name__, pymlu.__version__))
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


# TODO: adict can also be a list of dicts, see get_configs()
def dict_to_bunch(adict):
    # Lazy import
    from sklearn.utils import Bunch

    def bunchi(item):
        b = Bunch()
        b.update(item)
        return b

    return json.loads(json.dumps(adict), object_hook=lambda item: bunchi(item))


# Ref.: https://bit.ly/3qMjJF8
def dict_to_namespace(adict):
    return json.loads(json.dumps(adict),
                      object_hook=lambda item: SimpleNamespace(**item))


# TODO (IMPORTANT): in a single function with get_default_config_dict
def get_config_dict(cfg_type='main'):
    if cfg_type == 'main':
        cfg_filepath = get_main_config_filepath()
    elif cfg_type == 'log':
        cfg_filepath = get_logging_filepath()
    else:
        raise ValueError(f"Invalid cfg_type: {cfg_type}")
    return load_cfg_dict(cfg_filepath, cfg_type)


def get_config_from_locals(config, locals_dict, ignored_keys=None):
    locals_dict_copy = locals_dict.copy()
    default_ignored_keys = {'args', 'kwargs', 'ipdb', 'self'}
    if ignored_keys:
        ignored_keys = set(ignored_keys)
        ignored_keys.update(default_ignored_keys)
    else:
        ignored_keys = default_ignored_keys
    for k in ignored_keys:
        if locals_dict_copy.get(k):
            del locals_dict_copy[k]
    if config:
        new_config = {}
        locals_keys = locals_dict_copy.keys()
        for k in locals_keys:
            if k in config:
                new_config[k] = config[k]
            else:
                new_config[k] = locals_dict_copy[k]
        cfg = dict_to_bunch(new_config)
    else:
        cfg = dict_to_bunch(locals_dict_copy)
    return cfg


def get_configs(main_config_dict=None, model_configs=None, quiet=False,
                verbose=False, logging_level=None, logging_formatter=None):
    # Update default config dict
    cfgs = update_default_config(main_config_dict, model_configs)
    if quiet is None and cfgs:
        quiet = cfgs[0]['quiet']
    if verbose is None and cfgs:
        verbose = cfgs[0]['verbose']
    setup_log(use_default_log=True, quiet=quiet, verbose=verbose,
              logging_level=logging_level.upper(),
              logging_formatter=logging_formatter)
    return dict_to_bunch(cfgs)


def get_default_config_dict(cfg_type='main'):
    if cfg_type == 'main':
        cfg_filepath = get_default_main_config_filepath()
    elif cfg_type == 'log':
        cfg_filepath = get_default_logging_filepath()
    else:
        raise ValueError(f"Invalid cfg_type: {cfg_type}")
    return load_cfg_dict(cfg_filepath, cfg_type)


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
                               ignore_fnames=None, lower_model_names=False):
    categories = [] if categories is None else categories
    model_names = [] if model_names is None else model_names
    ignore_fnames = [] if ignore_fnames is None else ignore_fnames
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
            current_short_name = get_short_model_name(current_model_name)
            current_model_type = os.path.basename(path)
            current_model_category = os.path.basename(os.path.dirname(path))
            # TODO: in a function
            name_found = ''
            for name in model_names:
                cur_name = current_model_name.lower() if lower_model_names else current_model_name
                cur_short = current_short_name.lower() if lower_model_names else current_short_name
                name = name.lower() if lower_model_names else name
                if cur_name == name or cur_short == name:
                    name_found = name
                    model_names.remove(name)
                    break
            if name_found and model_type:
                # TODO (IMPORTANT): mention assert in notes
                assert model_type == current_model_type, \
                    f"Trying to train a model ({current_model_name}) that is " \
                    "different from the specified model type ({model_type})"
            if name_found or (current_model_type==model_type and
                              current_model_category in categories) or \
                (model_type is None and current_model_category in categories):
                # Add the fname since it is a valid one
                filepaths.append({'model_name': name_found,
                                  'filepath': os.path.join(path, fname)})
            else:
                # fname not part of a valid dirname (category) or not correct
                # model type or model name not found in the fname; next fname
                continue
    return filepaths, model_names


def get_settings(conf, cfg_type):
    if cfg_type == 'log':
        set_logging_field_width(conf['logging'])
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


def get_short_model_name(model_name):
    def get_acronym(compound_word):
        return ''.join([l for l in compound_word if not l.islower()])

    if DEFAULT_ABBREVIATIONS.get(model_name):
        short_name = DEFAULT_ABBREVIATIONS.get(model_name)
    else:
        acronym = get_acronym(model_name)
        short_name = acronym
    return short_name


def init_log(module__name__, module___file__=None, package_name=None):
    if module___file__:
        logger_ = logging.getLogger(get_logger_name(module__name__,
                                                    module___file__,
                                                    package_name))
    elif module__name__.count('.') > 1:
        logger_name = '.'.join(module__name__.split('.')[-2:])
        logger_ = logging.getLogger(logger_name)
    else:
        logger_ = logging.getLogger(module__name__)
    logger_.addHandler(NullHandler())
    return logger_


# TODO: not used
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


def list_model_info(use_cwd=True, show_all=True):
    msgs = []
    abbr_dict = {}
    if show_all:
        title = "***List of model categories and names***"
    else:
        title = "***List of model categories***"
    if use_cwd:
        source_msg = "Source: current working directory"
    else:
        source_msg = f"Source: {pymlu.__name__}"
    msgs.append(title)
    msgs.append(source_msg)
    acronyms = []
    module_found = False
    for i, module in enumerate(SKLEARN_MODULES, start=1):
        if i == 1:
            msgs.append("")
        if use_cwd:
            # Path to the model configs folder in the working directory
            module_dirpath = os.path.join(get_model_configs_dirpath(), module)
        else:
            # Path to the default model configs folder
            module_dirpath = os.path.join(get_default_model_configs_dirpath(),
                                          module)
        if os.path.exists(module_dirpath):
            spaces = '  ' if i < 10 else ' '
            # e.g. (3)  ensemble
            msgs.append(f"({i}){spaces}{module}")
            module_found = True
        else:
            continue
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
                            msgs.append(f"\t* {model_type}")
                        short_name = get_short_model_name(model_name)
                        msgs.append(f"\t    - {model_name} [{short_name}]")
                        abbr_dict.setdefault(short_name.lower(), model_name)
            msgs.append("")
    if show_all and module_found:
        msg = "\nNotes:\n- Beside each number in parentheses, it is the " \
              "model category\n- Between brackets, it is the model name " \
              "abbreviation\n"
        msgs.append(msg)
        msgs.append(source_msg)
    print_and_wait(msgs)
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


def namespace_to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        adict = vars(ns)
    else:
        adict = ns
    for k, v in adict.items():
        if isinstance(v, SimpleNamespace):
            v = vars(v)
            adict[k] = v
        if isinstance(v, dict):
            namespace_to_dict(v)
    return adict


def override_default_cfg_dict(new_cfg_dict, cfg_type):
    log_msgs = {'main': [], 'log': []}
    # Get default cfg dict
    if cfg_type == 'main':
        default_cfg_dict = load_cfg_dict(get_main_config_filepath(), cfg_type)
    elif cfg_type == 'log':
        # cfg_type = 'log'
        default_cfg_dict = load_cfg_dict(get_logging_filepath(), cfg_type)
    else:
        # TODO: raise AssertionError (check other places)
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


def print_and_wait(msgs, n_lines=25):
    idx_start = 0
    try:
        while True:
            group_msgs = msgs[idx_start:idx_start + n_lines]
            idx_start += n_lines
            for msg in group_msgs:
                print(msg)
            if idx_start >= len(msgs):
                break
            input("Press ENTER to continue (or Ctrl+C to exit)...")
            # Ref.: https://stackoverflow.com/a/52590238/14664104
            sys.stdout.write('\x1b[1A')
            sys.stdout.write('\x1b[2K')
    except KeyboardInterrupt:
        print("")


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


# TODO: specify log_dict change inline
def set_logging_field_width(log_dict):
    names = log_dict['loggers'].keys()
    if sys.argv and os.path.basename(sys.argv[0]) == 'mlearn':
        names = [n for n in names if not n.startswith('default_')]
    size_longest_name = len(max(names, key=len))
    for k, v in log_dict['formatters'].items():
        try:
            # TODO: add auto_field_width at the top
            v['format'] = v['format'].format(auto_field_width=size_longest_name)
        except KeyError:
            continue


def set_logging_formatter(log_dict, handler_names, formatter='simple'):
    # TODO: assert hander_names and formatter
    for handler_name in handler_names:
        log_dict['handlers'][handler_name]['formatter'] = formatter


def set_logging_level(log_dict, handler_names=None, logger_names=None,
                      level='DEBUG'):
    # TODO: assert handler_names, logger_names and level
    handler_names = handler_names if handler_names else []
    logger_names = logger_names if logger_names else []
    keys = ['handlers', 'loggers']
    for k in keys:
        for name, val in log_dict[k].items():
            if (not handler_names and not logger_names) or \
                    (k == 'handlers' and name in handler_names) or \
                    (k == 'loggers' and name in logger_names):
                val['level'] = level


def setup_log(use_default_log=False, quiet=False, verbose=False,
              logging_level=None, logging_formatter=None):
    logging_level = logging_level.upper()
    package_path = os.getcwd()
    if use_default_log:
        log_filepath = get_default_logging_filepath()
        main_cfg_msg = f'Default config path: {get_default_main_config_filepath()}'
        main_log_msg = f'Default logging path: {log_filepath}'
    else:
        log_filepath = get_logging_filepath()
        main_cfg_msg = f"Main config path: {get_main_config_filepath()}"
        main_log_msg = f'Logging path: {log_filepath}'
    # Get logging cfg dict
    log_dict = load_cfg_dict(log_filepath, cfg_type='log')
    # NOTE: if quiet and verbose are both activated, only quiet will have an effect
    # TODO: get first cfg_dict to setup log (same in train_models.py)
    if not quiet:
        if verbose:
            set_logging_level(log_dict, level='DEBUG')
        if logging_level:
            # TODO: add console_for_users at the top
            set_logging_level(log_dict, handler_names=['console_for_users'],
                              logger_names=['data'], level=logging_level)
        if logging_formatter:
            set_logging_formatter(log_dict, handler_names=['console_for_users'],
                                  formatter=logging_formatter)
        # Load logging config dict
        logging.config.dictConfig(log_dict)
    # =============
    # Start logging
    # =============
    logger.info("Running {} v{}".format(pymlu.__name__, pymlu.__version__))
    logger.info("Verbose option {}".format(
        "enabled" if verbose else "disabled"))
    logger.debug("Working directory: {}".format(package_path))
    logger.debug(main_cfg_msg)
    logger.debug(main_log_msg)


def update_default_config(new_data, model_configs=None):
    model_configs = model_configs if model_configs else []
    # TODO: show overriden options like with 'mlearn train'?
    # TODO: maybe instead explain that no because situation is different (we are working on kaggle)
    cfg_dicts = []
    # Get default main config dict
    default_cfg = get_default_config_dict()
    # Update default config dict wth new_data
    default_cfg.update(new_data)
    # Update default config dict with model_config_dicts
    if model_configs:
        for m_cfg in model_configs:
            # TODO: copy?
            default_cfg_copy = copy.deepcopy(default_cfg)
            default_cfg_copy.update({'model': m_cfg})
            cfg_dicts.append(default_cfg_copy)
    else:
        cfg_dicts.append(default_cfg)
    return cfg_dicts


# TODO (IMPORTANT): use a single function for all of these
# ------------------------------
# Default dirpaths and filepaths
# ------------------------------
def get_default_configs_dirpath():
    from pymlu.default_mlconfigs import __path__
    return __path__[0]


def get_default_logging_filepath():
    return os.path.join(get_default_configs_dirpath(), 'logging.py')


def get_default_main_config_filepath():
    return os.path.join(get_default_configs_dirpath(), 'config.py')


def get_default_model_configs_dirpath():
    # Path to the default model_configs directory
    return os.path.join(get_default_configs_dirpath(), MODEL_CONFIGS_DIRNAME)


def get_default_modules_dirpath():
    from pymlu.default_mlmodules import __path__
    return __path__[0]


# --------------------------
# CWD dirpaths and filepaths
# --------------------------
def get_configs_dirpath():
    from mlconfigs import __path__
    return __path__[0]


def get_logging_filepath():
    return os.path.join(get_configs_dirpath(), 'logging.py')


def get_main_config_filepath():
    return os.path.join(get_configs_dirpath(), 'config.py')


def get_model_configs_dirpath():
    # Path to the model_configs folder in the current working directory
    return os.path.join(os.getcwd(), CONFIGS_DIRNAME, MODEL_CONFIGS_DIRNAME)


def get_modules_dirpath():
    # Path to the module folder in the current working directory
    return os.path.join(os.getcwd(), MODULES_DIRNAME)


def get_module_filepath(module_name):
    assert module_name in MODULE_NAMES, f"Invalid module name: {module_name}"
    return os.path.join(get_modules_dirpath(), module_name+'.py')
