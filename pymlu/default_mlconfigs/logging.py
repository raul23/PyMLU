logging = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters":
    {
        "console":
        {
          "format": "[%(name)-30s] | %(levelname)-8s | %(message)s"
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
          "formatter": "simple"
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
        "data":
        {
          "level": "DEBUG",
          "handlers": ["console_only_msg"],
          "propagate": False
        },
        "default_mlmodules.explore_data":
        {
          "level": "DEBUG",
          "handlers": ["console"],
          "propagate": False
        },
        "default_mlmodules.train_models":
        {
          "level": "DEBUG",
          "handlers": ["console"],
          "propagate": False
        },
        "mlmodules.explore_data":
        {
          "level": "DEBUG",
          "handlers": ["console"],
          "propagate": False
        },
        "mlmodules.train_models":
        {
          "level": "DEBUG",
          "handlers": ["console"],
          "propagate": False
        },
        "pymlu.dautils":
        {
          "level": "DEBUG",
          "handlers": ["console"],
          "propagate": False
        },
        "pymlu.genutils":
        {
          "level": "DEBUG",
          "handlers": ["console"],
          "propagate": False
        },
        "pymlu.mlutils":
        {
          "level": "DEBUG",
          "handlers": ["console"],
          "propagate": False
        },
        "scripts.mlearn":
        {
          "level": "DEBUG",
          "handlers": ["console"],
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
