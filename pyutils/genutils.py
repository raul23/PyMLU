"""General utilities
"""
import argparse
import codecs
import importlib
import json
import logging.config
import os
import sys
from collections import OrderedDict
from logging import NullHandler
from runpy import run_path

# import ipdb
import pyutils


def get_short_logger_name(name):
    return '.'.join(name.split('.')[-2:])


logger = logging.getLogger(get_short_logger_name(__name__))
logger.addHandler(NullHandler())

logger_data = logging.getLogger('data')
logger_data.addHandler(NullHandler())


class ConfigBoilerplate:

    # eg. module_file = 'train_model.py'
    def __init__(self, module_file):
        # e.g. _module_file = 'train_model'
        self._module_file = os.path.basename(os.path.splitext(module_file)[0])
        # e.g. _package_name = 'titanic'
        self._package_name = os.path.basename(os.getcwd())
        self._package_path = os.getcwd()
        self.module = importlib.import_module(self._module_file)
        self._module_logger = self.module.logger
        self._module_name = self.module.__name__
        # =============================================
        # Parse command-line arguments and setup config
        # =============================================
        self._overridden_cfgs = {'cfg': [], 'log': []}
        retval = self._parse_cmdl_args()
        self.cfg_filepath = retval['cfg_filepath']
        self.log_filepath = retval['log_filepath']
        self.cfg_dict = retval['cfg_dict']
        self.log_dict = retval['log_dict']
        # ==============================
        # Logging setup from config file
        # ==============================
        self._setup_log_from_cfg()
        self._log_overridden_cfgs()

    def get_cfg_dict(self):
        return self.cfg_dict

    def get_logger(self):
        return self.module_logger

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

    def _parse_cmdl_args(self):
        cfg_data = {'cfg_filepath': None, 'log_filepath': None,
                    'cfg_dict': None, 'log_dict': None}
        parser = setup_argparser()
        args = parser.parse_args()
        if os.path.isdir(args.cfg_filepath) and not args.model:
            raise RuntimeError("Config directory provided but model (-m) "
                               "argument missing")
        if args.model:
            args.cfg_filepath = os.path.join(args.cfg_filepath, args.model + '_config.py')
        cfg_data['cfg_filepath'], cfg_data['log_filepath'] = get_cfg_filepaths(args)
        # Get config dict
        cfg_data['cfg_dict'] = load_cfg_dict(cfg_data['cfg_filepath'])
        # Get logging cfg dict
        cfg_data['log_dict'] = load_cfg_dict(cfg_data['log_filepath'],
                                             is_logging=True)
        # Override default cfg dict with user-defined cfg dict
        self._override_default_cfg_dict(cfg_data['cfg_dict'])
        # Override default logging cfg with user-defined logging cfg dict
        self._override_default_cfg_dict(cfg_data['log_dict'], 'log')
        return cfg_data

    # TODO: cfg_type = {'cfg', 'log'}
    def _override_default_cfg_dict(self, new_cfg_dict, cfg_type='cfg'):
        # Get default cfg dict
        if cfg_type == 'cfg':
            default_cfg_dict = load_cfg_dict(get_default_cfg_filepath())
        else:
            # cfg_type = 'log'
            filepath = get_default_logging_filepath()
            default_cfg_dict = load_cfg_dict(get_default_logging_filepath(),
                                             is_logging=True)
        for k, v in default_cfg_dict.items():
            if new_cfg_dict.get(k) is None:
                new_cfg_dict[k] = v
            else:
                if new_cfg_dict[k] != v:
                    if len(f"{v}") > 65 or len(f"{new_cfg_dict[k]}") > 65:
                        log_msg = f"** {k} **:\n{v}\n| -> {new_cfg_dict[k]}"
                    else:
                        log_msg = f"** {k} **: {v} -> {new_cfg_dict[k]}"
                    self._overridden_cfgs[cfg_type].append(log_msg)
                    v = new_cfg_dict[k]

    def _setup_log_from_cfg(self):
        self.module_logger = logging.getLogger(
            get_logger_name(self._package_name,
                            self._module_name,
                            self._module_file))
        # NOTE: if quiet and verbose are both activated, only quiet will have an effect
        if self.cfg_dict['quiet']:
            # TODO: disable logging completely? even error messages?
            self.module_logger.disabled = True
        else:
            # Load logging config dict
            if self.cfg_dict['verbose']:
                set_logging_level(self.log_dict)
            logging.config.dictConfig(self.log_dict)
        # =============
        # Start logging
        # =============
        logger.info("Running {} v{}".format(pyutils.__name__, pyutils.__version__))
        logger.info("Using the dataset: {}".format(self._package_name))
        logger.debug("Package path: {}".format(self._package_path))
        logger.info("Verbose option {}".format(
            "enabled" if self.cfg_dict['verbose'] else "disabled"))


def get_cfg_filepaths(args):
    # Get default cfg filepaths
    default_cfg = get_default_cfg_filepath()
    default_log = get_default_logging_filepath()

    # Get config filepaths from args (i.e. command-line)
    cmdline_cfg = os.path.abspath(args.cfg_filepath) if args.cfg_filepath else None
    cmdline_log = os.path.abspath(args.log_filepath) if args.log_filepath else None

    # Get config filepaths from command line (if they are defined) or default ones
    cfg_filepath = cmdline_cfg if cmdline_cfg else default_cfg
    log_filepath = cmdline_log if cmdline_log else default_log

    return cfg_filepath, log_filepath


def get_default_cfg_filepath():
    from default_configs import __path__ as configs_path
    return os.path.join(configs_path[0], 'config.py')


def get_default_logging_filepath():
    from default_configs import __path__ as configs_path
    return os.path.join(configs_path[0], 'logging.py')


# TODO: module_file must be the filename (not whole filepath)
def get_logger_name(package_name, module_name, module_file):
    if module_name == '__main__' or not module_name.count('.'):
        logger_name = "{}.{}".format(
            package_name,
            os.path.splitext(module_file)[0])
    elif module_name.count('.') > 1:
        logger_name = '.'.join(module_name.split('.')[-2:])
    else:
        logger_name = module_name
    return logger_name


def get_settings(conf, is_logging=False):
    _settings = {}
    if is_logging:
        return conf['logging']
    for opt_name, opt_value in conf.items():
        if not opt_name.startswith('__') and not opt_name.endswith('__'):
            _settings.setdefault(opt_name, opt_value)
    return _settings


def load_cfg_dict(cfg_filepath, is_logging=False):
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


def set_logging_level(log_dict, level='DEBUG'):
    keys = ['handlers', 'loggers']
    for k in keys:
        for name, val in log_dict[k].items():
            val['level'] = level


def setup_argparser():
    """Setup the argument parser for the command-line script.

    TODO

    Returns
    -------
    parser : argparse.ArgumentParser
        Argument parser.

    """
    cfg_filepath = get_default_cfg_filepath()
    log_filepath = get_default_logging_filepath()
    # Setup the parser
    parser = argparse.ArgumentParser(
        # usage="%(prog)s [OPTIONS]",
        # prog=os.path.basename(__file__),
        description='''\
TODO\n''',
        # formatter_class=argparse.RawDescriptionHelpFormatter)
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # ===============
    # General options
    # ===============
    # TODO: package name too? instead of program name (e.g. train_model.py)
    parser.add_argument("--version", action='version',
                        version='%(prog)s {}'.format(pyutils.__version__))
    parser.add_argument(
        "-c", "--cfg-filepath", dest="cfg_filepath", default=cfg_filepath,
        help='''Filepath to the main configuration file (.py) or the directory
        containing the model configuration files (.py).''')
    parser.add_argument(
        "-l", "--log-filepath", dest="log_filepath", default=log_filepath,
        help='''Filepath to the logging configuration file (.py)''')
    parser.add_argument(
        "-m", "--model", dest="model",
        help='''Model type to use (e.g. linear_svc, log_reg, perceptron)''')
    return parser
