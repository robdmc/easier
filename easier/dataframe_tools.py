from typing import Union, List
import re


def slugify(vals: Union[str, List[str]], sep: str = '_'):
    """
    Creates slugs out of string inputs.
    """
    if isinstance(vals, str):
        str_input = True
        vals = [vals]
    else:
        str_input = False
    out = [re.sub(r'[^A-Za-z0-9]+', sep, v.strip()).lower() for v in vals]
    out = [re.sub(r'_{2:}', sep, v) for v in out]
    out = [re.sub(r'^_', '', v) for v in out]
    out = [re.sub(r'_$', '', v) for v in out]

    if str_input:
        return out[0]
    else:
        return out
