# Checks the --config_file TOML against the script's argparse options before the values are used.
#
# read_config_from_file hands the TOML to argparse as a ready-made namespace, and argparse neither rejects names it does not
# know nor converts values it did not parse itself: a typo such as "moss_sigma_cutof = 0.9" was silently ignored, and
# "learning_rate = '1e-4'" or "use_zero_cond_dropout = 'false'" (a truthy string) reached the code as written. Every value
# now goes through the option it names, as if it had been given on the command line:
# - keys that name no option are listed in one warning with the closest option names and kept, as before (other tools,
#   e.g. the GUI, write their own keys into these files); --strict_config_keys makes them an error;
# - an option alias ("learning_rate_dora") is stored under the option's dest ("learning_rate_DoRA"); two keys for one
#   option are an error;
# - store_true / store_false / type=bool options need a TOML boolean;
# - int / float / str options convert like the command line, plus an integral float for an int and a number for a string;
#   anything else (a boolean for a number, 2.5 for an int, an array for a single value) is an error;
# - other argparse types (custom functions, Path) convert strings; other values reach the code unchanged, as before;
# - list options (nargs) take a TOML array, and a single value becomes a one-item list, as on the command line;
# - choices are checked after the conversion.
# All errors are reported together, with the file and line.

import argparse
import difflib
import re
from typing import Any, Dict, List, Optional

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

_BOOL_ACTIONS = (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction)
_BOOL_HINT = "needs true or false (a TOML boolean, without quotes)"
UNKNOWN_KEY_MODES = ("warn", "error", "ignore")


class ConfigValidationError(ValueError):
    pass


def option_names(parser: argparse.ArgumentParser) -> Dict[str, argparse.Action]:
    """Every name a TOML key may use for an option: its dest and its long option strings without dashes."""
    names: Dict[str, argparse.Action] = {}
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction) or action.dest == argparse.SUPPRESS:
            continue
        names.setdefault(action.dest, action)
        for option in action.option_strings:
            if option.startswith("--"):
                names.setdefault(option[2:], action)
    return names


def _kind(kind) -> str:
    return {int: "an integer", float: "a number", str: "a string"}.get(kind, getattr(kind, "__name__", repr(kind)))


def _convert_scalar(action: argparse.Action, value: Any) -> Any:
    kind = action.type
    if kind is bool:  # type=bool: on the command line any non-empty string, "False" included, is True
        if isinstance(value, bool):
            return value
        raise ValueError(_BOOL_HINT)
    if kind in (int, float, str):
        if isinstance(value, bool):
            raise ValueError(f"needs {_kind(kind)}, got a boolean")
        if kind is str:
            if isinstance(value, (str, int, float)):
                return value if isinstance(value, str) else str(value)
        elif isinstance(value, str):
            try:
                return kind(value.strip())
            except ValueError:
                raise ValueError(f"needs {_kind(kind)}") from None
        elif kind is int:
            if isinstance(value, int):
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
        elif isinstance(value, (int, float)):
            return float(value)
        raise ValueError(f"needs {_kind(kind)}")
    if kind is not None and isinstance(value, str):
        return kind(value)  # custom argparse type, as on the command line
    return value


def _check_choices(action: argparse.Action, items: List[Any]) -> None:
    if action.choices is None:
        return
    for item in items:
        if item not in action.choices:
            raise ValueError(f"{item!r} is not one of {list(action.choices)}")


def convert_value(action: argparse.Action, value: Any) -> Any:
    """The value the option would get from the command line; raises ValueError (or the type's own error)."""
    if isinstance(action, _BOOL_ACTIONS):
        if isinstance(value, bool):
            return value
        raise ValueError(_BOOL_HINT)
    if isinstance(action, (argparse._StoreAction, argparse._AppendAction)):
        is_list = isinstance(action, argparse._AppendAction) or action.nargs not in (None, argparse.OPTIONAL)
        if not is_list:
            converted = _convert_scalar(action, value)
            _check_choices(action, [converted])
            return converted
        items = [_convert_scalar(action, item) for item in (value if isinstance(value, (list, tuple)) else [value])]
        _check_choices(action, items)
        if isinstance(action.nargs, int) and not isinstance(action, argparse._AppendAction) and len(items) != action.nargs:
            raise ValueError(f"needs {action.nargs} values, got {len(items)}")
        if action.nargs == argparse.ONE_OR_MORE and not items:
            raise ValueError("needs at least one value")
        return items
    return value  # store_const, count and custom actions: unchanged


def _line_of(text: Optional[str], key: str) -> Optional[int]:
    if not text:
        return None
    match = re.search(r"^[ \t]*[\"']?" + re.escape(key) + r"[\"']?[ \t]*=", text, re.MULTILINE)
    return text.count("\n", 0, match.start()) + 1 if match else None


def validate_config(config: Dict[str, Any], parser: argparse.ArgumentParser, source: str = "<config>",
                    text: Optional[str] = None, unknown: str = "warn") -> Dict[str, Any]:
    """Return ``config`` (flattened TOML) with every key resolved to its option's dest and every value converted.

    Raises ConfigValidationError listing every error. Keys that name no option are kept unchanged and, by ``unknown``,
    reported in one warning ("warn"), raised with the errors ("error") or skipped silently ("ignore", for tools that read a
    training config with a smaller parser)."""
    if unknown not in UNKNOWN_KEY_MODES:
        raise ValueError(f"unknown must be one of {UNKNOWN_KEY_MODES}, got {unknown!r}")
    names = option_names(parser)
    out: Dict[str, Any] = {}
    errors: List[str] = []
    unknown_keys: List[str] = []
    owner: Dict[str, str] = {}
    for key, value in config.items():
        line = _line_of(text, key)
        where = f"{source}:{line}" if line else source
        action = names.get(key)
        if action is None:
            close = difflib.get_close_matches(key, list(names), n=3, cutoff=0.75)
            close += [name for name in names if name.lower() == key.lower() and name not in close]
            unknown_keys.append(f"{where}: {key!r}" + (f" (did you mean {', '.join(close)}?)" if close else ""))
            out[key] = value
            continue
        if action.dest in owner:
            errors.append(f"{where}: {key!r} sets the same option as {owner[action.dest]!r}")
            continue
        owner[action.dest] = key
        try:
            out[action.dest] = convert_value(action, value)
        except (ValueError, TypeError, argparse.ArgumentTypeError) as error:
            errors.append(f"{where}: {key} = {value!r}: {error}")

    if unknown_keys and unknown == "error":
        errors = [f"{entry}: unknown key" for entry in unknown_keys] + errors
    elif unknown_keys and unknown == "warn":
        logger.warning(
            f"{len(unknown_keys)} config keys name no option of this script and are ignored (typo?) /"
            " 設定ファイルの不明なキー (--strict_config_keys makes this an error):\n  " + "\n  ".join(unknown_keys)
        )
    if errors:
        raise ConfigValidationError("config file problems / 設定ファイルの問題:\n  " + "\n  ".join(errors))
    return out
