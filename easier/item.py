from copy import deepcopy


class Item(object):
    """
    Item is a simple container class that sets its attributes from constructor
    kwargs.  It supports both object and dictionary access to its attributes.
    So, for example, all of the following statements are supported.

    If the value associated with a key is a dict, that value will recursively
    be transformed into an Item.

    .. code-block:: python

       item = Item(a=1, b=2)
       item['c'] = 2
       a = item['a']
       item_dict = item.as_dict()
    """
    # I'm using unconventional "_item_self_" name here to avoid
    # conflicts when kwargs actually contain a "self" arg.

    def __init__(_item_self, **kwargs):
        for key, val in kwargs.items():
            # Recursively create items for dict members
            if isinstance(val, dict):
                val = Item(**val)
            _item_self[key] = val

    def __str__(_item_self):
        quoted_keys = [
            '\'{}\''.format(k) for k in sorted(vars(_item_self).keys())]
        att_string = ', '.join(quoted_keys)
        return 'Item({})'.format(att_string)

    def __repr__(_item_self):
        return _item_self.__str__()

    def __setitem__(_item_self, key, value):
        setattr(_item_self, key, value)

    def __getitem__(_item_self, key):
        return getattr(_item_self, key)

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
