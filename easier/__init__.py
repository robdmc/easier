# flake8: noqa
from .api import API, Importable

# This object is loaded with descriptors that lazy import items from modules
api_obj = API()


class Tools:
    def __init__(self):
        for att_name, att_obj in API.__dict__.items():
            if isinstance(att_obj, Importable):
                setattr(self, att_name, getattr(api_obj, att_name))


tools = Tools()


# This list will hold all the items the package will export
__all__ = []

# We cycle over the api items and add their names to the exportable namespace
for att_name, att_obj in tools.__dict__.items():
    __all__.append(att_name)
    cmd = f"{att_name} = getattr(tools, '{att_name}')"
    exec(cmd)


# (I)Python looks at this function to determine what is exported
def __dir__():
    return __all__
