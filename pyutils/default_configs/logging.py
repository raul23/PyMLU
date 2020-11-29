logging = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters":
    {
        "console":
        {
          "format": "[%(name)-20s] | %(levelname)-8s | %(message)s"
        },
        "only_msg":
        {
          "format": "%(message)s"
        },
        "simple":
        {
          "format": "%(levelname)-8s %(message)s"
        },
        "verbose":
        {
          "format": "%(asctime)s | %(name)-42s | %(levelname)-8s | %(message)s"
        }
    },

    "handlers":
    {
        "console":
        {
          "level": "INFO",
          "class": "logging.StreamHandler",
          "formatter": "console"
        },
        "console_only_msg":
        {
          "level": "INFO",
          "class": "logging.StreamHandler",
          "formatter": "only_msg"
        },
        "file":
        {
          "level": "INFO",
          "class": "logging.FileHandler",
          "filename": "debug.log",
          "mode": "a",
          "formatter": "simple",
          "delay": True
        }
    },

    "loggers":
    {
        "titanic.explore_data":
        {
          "level": "DEBUG",
          "handlers": ["console"],
          "propagate": False
        },
        "titanic.train_model":
        {
          "level": "DEBUG",
          "handlers": ["console"],
          "propagate": False
        },
        "pyutils.dautils":
        {
          "level": "DEBUG",
          "handlers": ["console"],
          "propagate": False
        },
        "pyutils.genutils":
        {
          "level": "DEBUG",
          "handlers": ["console"],
          "propagate": False
        },
        "pyutils.mlutils":
        {
          "level": "DEBUG",
          "handlers": ["console"],
          "propagate": False
        },
        "data":
        {
          "level": "DEBUG",
          "handlers": ["console_only_msg"],
          "propagate": False
        }
    },

    "root":
    {
        "level": "INFO",
        "handlers": ["console"],
        "propagate": False
    }
}
