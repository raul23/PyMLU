logging = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters":
    {
        "console":
        {
          "format": "%(name)-{auto_field_width}s | %(levelname)-8s | %(message)s"
        },
        "console_time":
        {
          "format": "%(asctime)s | %(levelname)-8s | %(message)s"
        },
        "only_msg":
        {
          "format": "%(message)s"
        },
        "simple":
        {
          "format": "%(levelname)-8s %(message)s"
        },
        "simple2":
        {
          "format": "%(levelname)-8s | %(message)s"
        },
        "verbose":
        {
          "format": "%(asctime)s | %(name)-{auto_field_width}s | %(levelname)-8s | %(message)s"
        }
    },

    "handlers":
    {
        # --------------------
        # User-defined handler
        # --------------------
        "console_for_users":
        {
          "level": "INFO",
          "class": "logging.StreamHandler",
          "formatter": "only_msg"
        },
        # ------------------
        # Debugging handlers
        # ------------------
        "console":
        {
          "level": "WARNING",
          "class": "logging.StreamHandler",
          "formatter": "only_msg"
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
        # ---------------------------------------
        # Loggers using console_for_users handler
        # ---------------------------------------
        "default_mlmodules.explore_data":
        {
          "level": "DEBUG",
          "handlers": ["console_for_users"],
          "propagate": False
        },
        "default_mlmodules.train_models":
        {
          "level": "DEBUG",
          "handlers": ["console_for_users"],
          "propagate": False
        },
        "mlmodules.explore_data":
        {
          "level": "DEBUG",
          "handlers": ["console_for_users"],
          "propagate": False
        },
        "mlmodules.train_models":
        {
          "level": "DEBUG",
          "handlers": ["console_for_users"],
          "propagate": False
        },
        "pymlu.dautils":
        {
          "level": "DEBUG",
          "handlers": ["console_for_users"],
          "propagate": False
        },
        "pymlu.mlutils":
        {
          "level": "DEBUG",
          "handlers": ["console_for_users"],
          "propagate": False
        },
        "scripts.mlearn":
        {
          "level": "DEBUG",
          "handlers": ["console_for_users"],
          "propagate": False
        },
        # ---------------------------------------------------
        # Loggers using console and console_only_msg handlers
        # ---------------------------------------------------
        "data":
        {
          "level": "INFO",
          "handlers": ["console_only_msg"],
          "propagate": False
        },
        "data_mlearn":
        {
          "level": "DEBUG",
          "handlers": ["console_only_msg"],
          "propagate": False
        },
        "pymlu.genutils":
        {
          "level": "DEBUG",
          "handlers": ["console"],
          "propagate": False
        },
    },

    "root":
    {
        "level": "INFO",
        "handlers": ["console"],
        "propagate": False
    }
}
