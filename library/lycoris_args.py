# LyCORIS reads a few network_args without parsing booleans: kohya.py passes decompose_both raw (the string "False" is
# truthy, so low-rank LoKr factorizes w1 as well), calls bool() on ggpo_conv, and or-chains the use_tucker aliases (so
# use_conv_cp=False shadows use_tucker=True). network_args always arrive as strings; turn these into real booleans first.

from typing import Dict

LYCORIS_RAW_BOOL_KEYS = ("decompose_both", "ggpo_conv", "use_tucker", "use_conv_cp", "use_cp", "disable_conv_cp")
_TRUE = ("true", "1", "yes", "on")
_FALSE = ("false", "0", "no", "off", "")


def fix_lycoris_bool_kwargs(network_module: str, net_kwargs: Dict[str, object]) -> Dict[str, object]:
    """Return net_kwargs with LyCORIS' unparsed boolean args as bool. Other network modules are left alone."""
    if not str(network_module or "").startswith("lycoris"):
        return net_kwargs
    fixed = dict(net_kwargs)
    for key in LYCORIS_RAW_BOOL_KEYS:
        value = fixed.get(key)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in _TRUE:
                fixed[key] = True
            elif text in _FALSE:
                fixed[key] = False
            else:
                raise ValueError(f"network_args {key}={value!r}: expected True or False")
    return fixed
