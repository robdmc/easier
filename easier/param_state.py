try:
    from astropy.units.quantity import Quantity
except:
    pass


class ParamState(object):
    INITIAL_VALUE = 0.

    def __init__(self, *args, **kwargs):
        """
        *args: a list of variable names
        **kwargs: any initializations to set initial arg values
        """
        # import here because I don't want the user to have to remember to import these
        from collections import OrderedDict
        from itertools import chain

        # allow variables to be passed in a single string
        if len(args) == 1 and isinstance(args[0], str):
            args = args[0].replace(',', ' ').split()

        # this dict will hold all variables
        self.vars = OrderedDict()

        # this keeps track of which variables are fixed
        self.fixed_vars = set()

        # a dict to hold the units for the args
        self.unit_dict = {}

        for arg in chain(args, kwargs.keys()):
            if arg not in self.vars:
                val = kwargs.get(arg, self.INITIAL_VALUE)
                try:
                    if isinstance(val, Quantity):
                        self.unit_dict[arg] = val.unit
                except NameError:
                    pass

                self.vars[arg] = val
                setattr(self, arg, val)

    def __setattr__(self, name, value):
        if hasattr(self, 'vars') and name in self.vars:
            self.vars[name] = value
        super(ParamState, self).__setattr__(name, value)

    def ingest(self, array):
        """
        Update the parameter state with values obtained from array
        """
        variables = [k for k in self.vars.keys() if k not in self.fixed_vars]
        if len(variables) != len(array):
            raise ValueError('Array to ingest should have length {}'.format(len(variables)))
        updates = dict(zip(variables, array))

        # attatch units if needed
        for var, val in updates.items():
            updates[var] = self.unit_dict.get(var, 1) * val

        self.vars.update(updates)
        self.__dict__.update(updates)

    def given(self, **kwargs):
        """
        This sets as constant the values of the supplied variables
        """
        bad_vars = set(kwargs.keys()) - set(self.vars.keys())
        if bad_vars:
            raise ValueError('Can only set these variables {}'.format(list(self.vars.keys())))
        self.vars.update(kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.fixed_vars = self.fixed_vars.union(set(kwargs.keys()))

    def __str__(self):
        """
        Prints a nice representation of the results
        """
        return self.df.to_string()

    def _repr_html_(self):
        # makes for nice display in notebook
        return self.df.to_html()

    def __repr__(self):
        return self.__str__()

    @property
    def df(self):
        # again, import here because don't want user to have to remember import
        import pandas as pd
        rec_list = []
        for k, v in self.vars.items():
            kind = '*' if k in self.fixed_vars else ''
            rec_list.append((k, v, kind))
        df = pd.DataFrame(rec_list, columns=['var', 'val', 'const'])
        df = df.sort_values(by=['const', 'var'])
        df = df.set_index('var')
        df.index.name = None
        return df

    @property
    def array(self):
        """
        Returns only the non-constant variables as an array.  Use this for
        creating the initial state
        """
        # import here so user doesn't have to
        import numpy as np
        out = []
        for key, val in self.vars.items():
            if key not in self.fixed_vars:
                try:
                    if isinstance(val, Quantity):
                        val = val.value
                except NameError:
                    pass
                out.append(val)

        return np.array(out)

    @property
    def tuple(self):
        """
        Return a tuple of all variables.  Use this for calling functions
        """
        return tuple(self.vars.values())
