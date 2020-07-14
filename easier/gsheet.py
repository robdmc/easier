import itertools
import json
import os
from textwrap import dedent
import string
from typing import Optional

import easier as ezr
import gspread
import pandas as pd
import numpy as np
import datetime

CONFIG_FILE_NAME = os.path.realpath(
    os.path.expanduser('~/.config/gspread/service_account.json')
)


class Email:
    def __init__(self, config_file_name):
        self.config_file_name = config_file_name

    @ezr.cached_property
    def email(self):
        with open(self.config_file_name) as buff:
            blob = json.load(buff)

        return blob['client_email']

    def __get__(self, obj, objclass):
        return self.email

    def __set__(self, obj, val):
        raise NotImplementedError('Can\'t set the email property')


class Example:
    def __get__(self, obj, objclass):
        example = dedent(f"""
            Share document with this email address:

                {objclass.email}

            Then:
            goog = ezr.GSheet(document_name, sheet_name)
            goog.store_frame(edf)

        """)
        return example

    def __set__(self, obj, val):
        raise NotImplementedError('Can\'t set the example property')


class GSheet:
    email = Email(CONFIG_FILE_NAME)
    example = Example()

    def __init__(self, doc, sheet):
        """
        Expects service account config at ~/.config/gspread/service_account.json

        See http://gspread.readthedocs.io/en/latest/oauth2.html for setting
        up sheets to enable api.
        """
        self._api = None
        self._document = None
        self._sheet = None
        self._df_cache = {}
        self.document = self.api.open(doc)
        self.sheet = self.document.worksheet(sheet)

    @classmethod
    def print_example(cls):
        print(cls.example)

    @ezr.cached_property
    def col_to_num(self):
        """
        Translate a spreadsheet column name to a column number
        """
        uppers = list(string.ascii_uppercase)
        return {
            val: ind + 1
            for (ind, val) in enumerate(''.join(list(t)) for t in list(itertools.product([''] + uppers, uppers)))}

    @ezr.cached_property
    def num_to_col(self):
        """
        Translate a column number to a spreadsheet column name
        """
        return {num: col for (col, num) in self.col_to_num.items()}

    @property
    def api(self):
        gc = gspread.service_account()
        return gc

    @property
    def keyfile_dict(self):
        # Initialize the key file dict
        kf_dict = {}

        # Populate the keyfile params from the env
        for env_var in self.ENV_VARS:
            key = env_var.replace('GOOGLE_', '').lower()
            if env_var not in os.environ:
                raise ValueError(f'Environment variable {env_var} is not defined')
            kf_dict[key] = os.environ[env_var]

        # The private key might have unescaped newlines.  Change those to real newlines
        kf_dict['private_key'] = kf_dict['private_key'].replace('\\n', '\n')
        return kf_dict

    def to_dataframe(self, header_row=1, reload=True):
        """
        Dump the current sheet to a dataframe using the specified row
        as a source for column names
        """
        # get a key to identify this frame in the cache
        key = (self.document.title, self.sheet.title)

        # fill cache if needed and returned cached frame
        df = self._df_cache.get(key)
        if df is None or reload:
            recs = self.sheet.get_all_records(head=header_row)
            columns = self.sheet.row_values(header_row)
            self._df_cache[key] = pd.DataFrame(recs)[columns]
        return self._df_cache[key].copy()

    def write_cell(self, coord, value):
        self.write_cells([(coord, value)])

    def write_cells(self, tuples):
        """
        Tuples of (coord, value)
        where coord is something like 'C15'
        """
        cells = []
        for (coord, val) in tuples:
            cell = self.sheet.acell(coord)
            cell.value = val
            cells.append(cell)

        self.sheet.update_cells(cells)

    def store_frame_to_coords(
            self,
            df: pd.DataFrame,
            top_left_coord: str,
            clear_to_bottom: bool = False,
            max_num_rows: Optional[int] = None,
            max_num_cols: Optional[int] = None
    ):
        """
        Args:
            df: the dataframe to store
            top_left_coord: The spreadsheet coordinate (e.g. 'A15') of the top left corner
            max_num_rows: Limit the dataframe to have this many rows before pushing
            max_num_cols: Limit the dataframe to have this mancy cols before pushing
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
                # This is a hack.  Dates will be exported as days from epoch
                # Format this as date to get the dates you want
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
        Clears the specified sheet and repopulates from specified dataframe
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
                # This is a hack.  Dates will be exported as days from epoch
                # Format this as date to get the dates you want
                if isinstance(value, pd.Timestamp):
                    value = (value.to_pydatetime() - gsheet_epoc).days
                elif value == '':
                    value = None
                elif type(value) in [np.int64, np.int32, np.int16, np.int8]:
                    value = int(value)
                cell.value = value
        self.sheet.update_cells(cells)
