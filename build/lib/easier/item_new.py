from copy import deepcopy


class Item(object):
    """
    A simple container class that sets its attributes from constructor kwargs.
    Supports both object and dictionary access to its attributes.

    Args:
        **kwargs: Arbitrary keyword arguments that will be set as attributes.
                 If a value is a dict, it will be recursively transformed into an Item.
    """

    # I'm using unconventional "_item_self_" name here to avoid
    # conflicts when kwargs actually contain a "self" arg.

    def __init__(_item_self, **kwargs):
        visible_attributes = []

        for key, val in kwargs.items():
            # Recursively create items for dict members
            if isinstance(val, dict):
                val = Item(**val)
            _item_self[key] = val
            visible_attributes.append(key)
        _item_self._visible_attributes = visible_attributes

    def __str__(_item_self):
        quoted_keys = [
            "'{}'".format(k)
            for k in sorted(vars(_item_self).keys())
            if k != "_visible_attributes"
        ]
        att_string = ", ".join(quoted_keys)
        return "Item({})".format(att_string)

    def __repr__(_item_self):
        return _item_self.__str__()

    def __setitem__(_item_self, key, value):
        setattr(_item_self, key, value)
        if key != "_visible_attributes":
            _item_self._visible_attributes.append(key)

    def __getitem__(_item_self, key):
        return getattr(_item_self, key)

    def __len__(self):
        return len(self._visible_attributes)

    def __dir__(self):
        return self._visible_attributes

    def as_dict(self, copy=False):
        if copy:
            return dict(**self.__dict__)
        else:
            return self.__dict__

    def to_dict(self, copy=False):
        return self.as_dict(copy=copy)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def clone(self):
        return deepcopy(self)
