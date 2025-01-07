from pathlib import Path
import yaml
import re
import os
from typing import Union
import logging.config
import socket
from datetime import datetime

ROOT = Path(__file__).resolve().parent
CFG_DIR = ROOT / ".." / "cfg"
LOGGING_NAME = "RAG-SQL-Agent"
VERBOSE = str(os.getenv("VERBOSE", True)).lower() == "true"  # global verbose mode
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def keys2attributes(dictionary):
    class DotAccessibleDict(dict):
        """
        A dictionary with attribute-style access.

        It maps dictionary keys to object attributes, allowing for direct key access using the 'dot'
        notation instead of the bracket notation typically used with dictionaries.
        An AttributeError is raised if attribute not found.

        Attributes:
            No public attributes are defined by this class.

        Methods:
            __getattr__(self, key):
            Retrieve the value associated with 'key' if it exists in the dictionary.
        """

        def __getattr__(self, key):
            if key in self:
                return self[key]
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            )

    o_dict = DotAccessibleDict(dictionary)
    return o_dict


def load_yaml(file: Union[str, Path] = "data.yaml") -> dict:
    """
    Loads a YAML file and returns its contents as a dictionary.

    This function reads from a YAML file, ensuring that the file exists and
    is in the correct '.yaml' format before loading it.
    Special characters in the file are removed (if any), and the YAML content
    is parsed into a dictionary. All keys in the loaded dictionary are formatted
    to be compatible as attribute names.

    Args:
        file (Union[str, Path], optional): The path to the YAML file. Defaults to "data.yaml".

    Returns:
        dict: A dictionary containing the parsed contents of the YAML file."""
    assert os.path.exists(file), f"File not found in path {file}"
    assert (isinstance(file, str) and file.endswith(".yaml")) or isinstance(
        file, Path
    ), "Not the proper format, must be a yaml file string variable"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string
        # Remove special characters
        if not s.isprintable():
            s = re.sub(
                r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+",
                "",
                s,
            )
        params = keys2attributes(yaml.safe_load(s))
    return params


class ConfigDict(dict):
    """
    A dictionary subclass that enables attribute-style access to its keys.

    This custom dictionary class throws a KeyError when a key is not found, similar
    to a standard dictionary. However, it allows accessing keys as attributes.

    Args:
        None

    Attributes:
        None

    Methods:
        register(obj=None, name=None): Register the given object under the given name or
            `obj.__name__`.
        __missing__(key): Returns KeyError when the given key is not found.
        __getattr__(name): Retrieves the value of the attribute with the specified name.
        __setattr__(name, value): Set the value of the attribute.
        __delattr__(name): Delete the specified attribute if it exists.

    Private Methods:
        _do_register(name, obj): Register the given object under the specified name.
    """

    def _do_register(self, name, obj):
        """
        Registers an object with a given name in the registry.

        This function adds an object to the registry if the name provided
        has not already been registered. An assertion will raise an error
        if an attempt is made to register an object with a duplicate name.

        Args:
            name (str): The name under which the object will be registered.
            obj (Any): The object to be registered.

        Returns:
            None: This function does not return a value.
        """
        assert name not in self, (
            f"An object named '{name}' was already registered "
            f"in '{self._name}' registry!"
        )
        self[name] = obj

    def register(self, obj=None, name=None):
        """
        Register the given object under the specified name or `obj.__name__`.

        This function can be used as either a decorator or a regular function call.
        When used as a decorator, it allows for easy registration of functions or
        classes under a specified name. If no name is provided, the object's name
        will be used.

        Args:
            obj (callable, optional): The object to be registered. If None, it
                indicates that the function is being used as a decorator.
            name (str, optional): The name under which to register the object.
                If not provided, `obj.__name__` will be used.

        Returns:
            callable: If used as a decorator, this returns the decorated function
            or class. If called with an object, it returns None.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                reg_name = name if name is not None else func_or_class.__name__
                self._do_register(reg_name, func_or_class)
                return func_or_class

            return deco
        # used as a function call
        reg_name = name if name is not None else obj.__name__
        self._do_register(reg_name, obj)

    def __missing__(self, key):
        """
        Raises a KeyError when the requested key is not found in the collection.

        This method is typically used in conjunction with __getitem__ to handle cases
        when a key is missing from a dictionary-like object. It ensures that any attempt
        to access a non-existent key will result in a KeyError being raised, facilitating
        error handling in the calling code.

        Args:
            key (hashable): The key that was not found.

        Returns:
            KeyError: Raises a KeyError with the specified missing key.
        """
        raise KeyError(key)

    def __getattr__(self, name):
        """
        Retrieves the value of the attribute with the specified name.

        This method allows dynamic access to the object's attributes by their name.
        If the attribute exists, its value is returned; otherwise, an AttributeError
        is raised.

        Args:
            name (str): The name of the attribute to retrieve.

        Returns:
            object: The value of the attribute with the specified name.

        Raises:
            AttributeError: If the attribute does not exist in the object.
        """
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            ) from e

    def __setattr__(self, name, value):
        """
        Set the value of an attribute in the object.

        This method allows you to set an attribute of the instance using the
        attribute's name as a string. It effectively enables dynamic
        attribute assignment within the object.

        Args:
            name (str): The name of the attribute to set.
            value (object): The value to assign to the attribute.

        Returns:
            None: This method does not return a value.
        """
        self[name] = value

    def __delattr__(self, name):
        """
        Delete the specified attribute from the object.

        This method attempts to delete the attribute with the given name.
        If the attribute does not exist, it raises an AttributeError with
        a message indicating that the attribute is not found.

        Args:
            name (str): The name of the attribute to delete.

        Returns:
            None: This function does not return a value.
        """
        try:
            del self[name]
        except KeyError as e:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            ) from e


def set_logging(name="app_logger", verbose=True):
    # Determine log directory one level up and ensure it exists
    log_dir = Path("/tmp/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate a timestamp for the log file name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"app_{timestamp}.log"

    # Get the server hostname
    hostname = socket.gethostname()

    # Configure logging
    rank = int(os.getenv("RANK", -1))
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(hostname)s - %(levelname)s - %(message)s",  # Add hostname placeholder here
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": level,
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(log_file),
                "maxBytes": 5 * 1024 * 1024,
                "backupCount": 3,
                "formatter": "default",
                "level": level,
            },
        },
        "loggers": {
            name: {
                "level": level,
                "handlers": ["console", "file"],
                "propagate": False,
            }
        },
    }

    # Apply the logging configuration
    logging.config.dictConfig(log_config)

    # Add the hostname dynamically to each log record
    logger = logging.getLogger(name)
    logger.addFilter(lambda record: setattr(record, "hostname", hostname) or record)


set_logging(LOGGING_NAME, verbose=VERBOSE)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)
