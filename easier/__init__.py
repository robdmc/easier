# flake8: noqa
from .api import API, Importable

# This object is loaded with descriptors that lazy import items from modules
api_obj = API()

# This list will hold all the items the package will export
dir_list = []


# We don't actually define any of the import in the namespace.
# We rely on this function to lazy load them
def __getattr__(name):
    return getattr(api_obj, name)


# We cycle over the api items and add their names to the exportable namespace
for att_name, att_obj in API.__dict__.items():
    if isinstance(att_obj, Importable):
        dir_list.append(att_name)


# (I)Python looks at this function to determine what is exported
def __dir__():
    return dir_list
