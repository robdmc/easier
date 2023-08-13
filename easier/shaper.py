class ListShaper:
    def flatten(self, shaped):
        import pandas as pd

        return pd.Series(shaped)

    def expand(self, flattened):
        return list(flattened)


class ArrayShaper:
    shape = None

    def flatten(self, shaped):
        import pandas as pd

        self.shape = shaped.shape
        return pd.Series(shaped.flatten())

    def expand(self, flattened):
        return flattened.values.reshape(self.shape)


class SeriesShaper:
    index = None

    def flatten(self, shaped):
        import pandas as pd

        self.index = shaped.index
        return pd.Series(shaped.values)

    def expand(self, flattened):
        import pandas as pd

        return pd.Series(
            flattened.values,
            index=self.index,
        )


class FrameShaper:
    index = None
    columns = None
    shape = None

    def flatten(self, shaped):
        import pandas as pd

        self.index = shaped.index
        self.columns = shaped.columns
        self.shape = shaped.values.shape

        return pd.Series(shaped.values.flatten())

    def expand(self, flattened):
        import pandas as pd

        return pd.DataFrame(
            flattened.values.reshape(self.shape), index=self.index, columns=self.columns
        )


class Shaper:
    """
    Datasets come in a variety of containers and sometimes you
    want to pass the data to a function that requires a one-dimenional
    container (usually an array).

    This class provides an easy method of transforming your data to
    and from a 1-dimentional series and back, preserving the containers
    shape (and columns/indexes for pandas objects.)

    The "flattened" data will always be a pandas series.

    The call signature is the same for all allowed datatypes.

    # Create a shaper
    shaper = ezr.Shaper()

    # Flatten the data
    flat_series = shaper.flatten(df)

    # Mutate the elements of the flat series
    # ** CAN NOT CHANGE THE LENGTH OR POSITION OF ELEMEMNTS **
    flat_series = process(flat_series)

    # Reconstitude the flat series to its original container
    # with possibly mutated elements
    df = shaper.expand(flat_series)
    """

    def __init__(self):
        import numpy as np
        import pandas as pd

        # This holds an instance of the appropriate shaper
        self.shaper = None

        # This hold a mapping of allowed types to the appropriate shaper class
        self.type_mapper = {
            list: ListShaper,
            np.ndarray: ArrayShaper,
            pd.Series: SeriesShaper,
            pd.DataFrame: FrameShaper,
        }

    def flatten(self, shaped):
        """
        Flatten a data container to a pandas Series
        """
        # Make sure only allowed types are used
        if type(shaped) not in self.type_mapper:
            msg = f"Allowed input types are {list(self.type_mapper.keys())}"
            raise ValueError(msg)

        # Create an instance of the proper shaper
        self.shaper = self.type_mapper[type(shaped)]()

        # Returned the flattened data
        return self.shaper.flatten(shaped)

    def expand(self, flattened):
        # Return the expanded data
        return self.shaper.expand(flattened)
