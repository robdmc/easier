from .utils import cached_property
from textwrap import dedent
from typing import Optional
import datetime
import gspread
import itertools
import json
import numpy as np
import os
import pandas as pd
import string

import itertools
import json
import os
from textwrap import dedent
import string
from typing import Optional
from .utils import cached_property
import pandas as pd
import numpy as np
import datetime
config_file_locations = [os.path.expanduser('/developer/.config/gspread/service_account.json'), os.path.expanduser('~/.config/gspread/service_account.json')]
for CONFIG_FILE_NAME in config_file_locations:
    if os.path.isfile(CONFIG_FILE_NAME):
        break

class Email:

    def __init__(self, config_file_name):
        self.config_file_name = config_file_name

    @cached_property
    def email(self):
        try:
            with open(self.config_file_name) as buff:
                blob = json.load(buff)
            return blob['client_email']
        except:
            return 'email_not_found'

    def __get__(self, obj, objclass):
        return self.email

    def __set__(self, obj, val):
        raise NotImplementedError("Can't set the email property")

class Example:

    def __get__(self, obj, objclass):
        example = dedent(f'\n            Share document with this email address:\n\n                {objclass.email}\n\n            Then:\n            goog = ezr.GSheet(document_name, sheet_name)\n            goog.store_frame(edf)\n\n        ')
        return example

    def __set__(self, obj, val):
        raise NotImplementedError("Can't set the example property")

class GSheet:
    """
    A class for interacting with Google Sheets using the Google Sheets API.

    This class provides methods to read from and write to Google Sheets,
    including functionality to work with pandas DataFrames.

    Attributes:
        email (str): The email address associated with the service account.
        example (str): An example showing how to use the GSheet class.
    """
    email = Email(CONFIG_FILE_NAME)
    example = Example()

    def __init__(self, doc, sheet):
        """
        Initialize a GSheet instance.

        Args:
            doc (str): The name or ID of the Google Sheets document.
            sheet (str): The name of the worksheet within the document.

        Raises:
            RuntimeError: If the service account configuration file is not found.
        """
        self._api = None
        self._document = None
        self._sheet = None
        self._df_cache = {}
        self.document = self.api.open(doc)
        self.sheet = self.document.worksheet(sheet)
        if not os.path.isfile(CONFIG_FILE_NAME):
            url = 'http://gspread.readthedocs.io/en/latest/oauth2.html'
            msg = f'\n\nFile does not exist:\n{CONFIG_FILE_NAME}\n\nSee {url}'
            raise RuntimeError(msg)

    @classmethod
    def print_example(cls):
        """
        Print an example showing how to use the GSheet class.
        """
        print(cls.example)

    @cached_property
    def col_to_num(self):
        """
        Create a mapping from spreadsheet column names to column numbers.

        Returns:
            dict: A dictionary mapping column names (e.g., 'A', 'B', 'AA') to their
                  corresponding column numbers (1-based).
        """
        uppers = list(string.ascii_uppercase)
        return {val: ind + 1 for ind, val in enumerate((''.join(list(t)) for t in list(itertools.product([''] + uppers, uppers))))}

    @cached_property
    def num_to_col(self):
        """
        Create a mapping from column numbers to spreadsheet column names.

        Returns:
            dict: A dictionary mapping column numbers (1-based) to their
                  corresponding column names (e.g., 'A', 'B', 'AA').
        """
        return {num: col for col, num in self.col_to_num.items()}

    @property
    def api(self):
        """
        Get the gspread API client.

        Returns:
            gspread.Client: An authenticated gspread client instance.
        """
        gc = gspread.service_account(CONFIG_FILE_NAME)
        return gc

    def to_dataframe_as_values(self):
        """
        Convert the entire sheet to a pandas DataFrame without header processing.

        Returns:
            pd.DataFrame: A DataFrame containing all values from the sheet.
        """
        lol = self.sheet.get_all_values()
        return pd.DataFrame(lol)

    def to_dataframe(self, header_row=1, reload=True):
        """
        Convert the sheet to a pandas DataFrame using the specified row as headers.

        Args:
            header_row (int, optional): The row number to use as column headers.
                                      Defaults to 1.
            reload (bool, optional): Whether to force reload the data from the sheet.
                                   Defaults to True.

        Returns:
            pd.DataFrame: A DataFrame containing the sheet data with proper column headers.
        """
        key = (self.document.title, self.sheet.title)
        df = self._df_cache.get(key)
        if df is None or reload:
            recs = self.sheet.get_all_records(head=header_row)
            columns = self.sheet.row_values(header_row)
            self._df_cache[key] = pd.DataFrame(recs)[columns]
        return self._df_cache[key].copy()

    def read_cell(self, coord):
        """
        Read the value of a specific cell.

        Args:
            coord (str): The cell coordinate (e.g., 'A1', 'B2').

        Returns:
            Any: The value of the specified cell.
        """
        cell = self.sheet.acell(coord, value_render_option='UNFORMATTED_VALUE')
        return cell.value

    def write_cell(self, coord, value):
        """
        Write a value to a specific cell.

        Args:
            coord (str): The cell coordinate (e.g., 'A1', 'B2').
            value (Any): The value to write to the cell.
        """
        self.write_cells([(coord, value)])

    def read_formula(self, coord):
        """
        Read the formula of a specific cell.

        Args:
            coord (str): The cell coordinate (e.g., 'A1', 'B2').

        Returns:
            str: The formula in the specified cell.
        """
        cell = self.sheet.acell(coord, value_render_option='FORMULA')
        return cell.value

    def write_formula(self, coord, value):
        """
        Write a formula to a specific cell.

        Args:
            coord (str): The cell coordinate (e.g., 'A1', 'B2').
            value (str): The formula to write to the cell.
        """
        self.sheet.update_acell(coord, value)

    def write_cells(self, tuples):
        """
        Write multiple values to multiple cells.

        Args:
            tuples (list): A list of (coord, value) tuples where coord is a cell
                          coordinate (e.g., 'C15') and value is the value to write.
        """
        cells = []
        for coord, val in tuples:
            cell = self.sheet.acell(coord)
            cell.value = val
            cells.append(cell)
        self.sheet.update_cells(cells)

    def store_frame_to_coords(self, df: pd.DataFrame, top_left_coord: str, clear_to_bottom: bool=False, max_num_rows: Optional[int]=None, max_num_cols: Optional[int]=None):
        """
        Store a DataFrame to a specific location in the sheet.

        Args:
            df (pd.DataFrame): The DataFrame to store.
            top_left_coord (str): The spreadsheet coordinate (e.g., 'A15') of the top left corner.
            clear_to_bottom (bool, optional): Whether to clear all cells below the data.
                                            Defaults to False.
            max_num_rows (int, optional): Limit the number of rows to write.
                                        Defaults to None.
            max_num_cols (int, optional): Limit the number of columns to write.
                                        Defaults to None.
        """
        top_left_coord = top_left_coord.upper()
        df = df.reset_index(drop=True).fillna('')
        if max_num_rows:
            df = df.iloc[:max_num_rows, :]
        if max_num_cols:
            df = df.iloc[:, :max_num_cols]
        cell = self.sheet.acell(top_left_coord)
        top_row = cell.row
        left_col = cell.col
        bottom_row = cell.row + len(df)
        right_col = cell.col + len(df.columns) - 1
        if clear_to_bottom:
            range_string = f"'{self.sheet.title}'!{top_left_coord}:{self.num_to_col[right_col]}"
            self.document.values_clear(range_string)
        else:
            range_string = f"'{self.sheet.title}'!{top_left_coord}:{self.num_to_col[right_col]}{bottom_row}"
            self.document.values_clear(range_string)
        cells = self.sheet.range(top_row, left_col, bottom_row, right_col)
        gsheet_epoc = datetime.datetime(1899, 12, 30)
        for cell in cells:
            frame_row = cell.row - top_row - 1
            frame_col = cell.col - left_col
            if cell.row == top_row:
                cell.value = list(df.columns)[frame_col]
            else:
                value = df.values[frame_row, frame_col]
                if isinstance(value, pd.Timestamp):
                    value = (value.to_pydatetime() - gsheet_epoc).days
                elif value == '':
                    value = None
                elif type(value) in [np.int64, np.int32, np.int16, np.int8]:
                    value = int(value)
                cell.value = value
        self.sheet.update_cells(cells)

    def store_frame(self, df, starting_row=1, total_rows=None, total_cols=None):
        """
        Clear the sheet and store a DataFrame starting from the specified row.

        Args:
            df (pd.DataFrame): The DataFrame to store.
            starting_row (int, optional): The row number to start writing from.
                                        Defaults to 1.
            total_rows (int, optional): The total number of rows to allocate.
                                      Defaults to None.
            total_cols (int, optional): The total number of columns to allocate.
                                      Defaults to None.
        """
        df = df.reset_index(drop=True).fillna('')
        self.sheet.clear()
        if total_rows is None:
            needed_row_count = len(df) + 20
        else:
            needed_row_count = total_rows
        if total_cols is None:
            needed_col_count = len(df.columns) + 20
        else:
            needed_col_count = total_cols
        self.sheet.resize(rows=needed_row_count, cols=needed_col_count)
        min_row = starting_row
        max_row = min_row + len(df)
        min_col = 1
        max_col = len(df.columns)
        cells = self.sheet.range(min_row, min_col, max_row, max_col)
        gsheet_epoc = datetime.datetime(1899, 12, 30)
        for cell in cells:
            if cell.row == starting_row:
                cell.value = list(df.columns)[cell.col - 1]
            else:
                value = df.values[cell.row - starting_row - 1, cell.col - 1]
                if isinstance(value, pd.Timestamp):
                    value = (value.to_pydatetime() - gsheet_epoc).days
                elif value == '':
                    value = None
                elif type(value) in [np.int64, np.int32, np.int16, np.int8]:
                    value = int(value)
                cell.value = value
        self.sheet.update_cells(cells)