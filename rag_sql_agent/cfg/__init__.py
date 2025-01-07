from rag_sql_agent.utils import load_yaml, CFG_DIR, LOGGER
from typing import List, Tuple, Dict
import sys
import contextlib
import re
from difflib import get_close_matches


def get_default_args(file="default.yaml"):
    """
    Loads default arguments from a YAML file.

    This function reads a YAML file, with an option to specify a different filename
    located within a configuration directory (CFG_DIR).
    The loaded arguments are returned as a dictionary.

    Args:
        file (str, optional): The filename of the YAML configuration file to be
        loaded (default is "default.yaml").

    Returns:
        dict: A dictionary containing the loaded configuration parameters.
    """
    args = load_yaml(CFG_DIR / file)
    return args


DEFAULT_CFG = get_default_args()


def parse_custom_args(args: List[str]) -> Dict[str, str]:
    """
    Parses custom arguments provided via the command line.

    Args:
        args (List[str]): List of arguments from the command line.

    Returns:
        dict: Dictionary of custom arguments where keys are argument names
              and values are the argument values.
    """
    custom_args = {}
    for arg in args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            custom_args[key] = value
        else:
            LOGGER.warning(f"Ignored malformed argument: {arg}")
    return custom_args


def entrypoint(debug=""):
    args = (debug.split(" ") if debug else sys.argv)[1:]

    # Predefined special commands
    special = {
        "help": lambda: LOGGER.info("Usage: <script> [key1=value1 key2=value2 ...]"),
    }

    # Parse custom arguments
    custom_args = parse_custom_args(args)

    # Merge default config with custom arguments
    config = {**DEFAULT_CFG, **custom_args}

    # Handle special commands if any
    for arg in args:
        if arg in special:
            special[arg]()
            return

    LOGGER.info("Final configuration: %s", config)

    from rag_sql_agent.data_pipelines.agent import LocalLLMAgent

    question = config["question"]
    key_to_remove = "question"
    if key_to_remove in config:
        del config[key_to_remove]
    else:
        print(f"Key '{key_to_remove}' not found.")
    LocalLLMAgent(question=question, **config)
