import abc
import os
import easier as ezr
from typing import Optional

# from google.cloud import bigquery
# from google.oauth2 import service_account
import pandas_gbq
import pandas as pd
import numpy as np

example = """
from easier.bigquery import BQAppender, BQTable, BQDataset
import pandas as pd
import datetime


class MyAppender(BQAppender):
    def get_dataframe(self, **kwargs):
        # self.validate_kwargs(**kwargs)

        df = pd.DataFrame(
            {
                'my_str': ['a', 'b'],
                'my_int': [1, 2],
                'my_float': [3., 4.],
            }
        )
        time = kwargs['time']
        df['time'] = time
        return df

    def validate_kwargs(self, **kwargs):
        if 'time' not in kwargs:
            raise ValueError('kwargs must contain "time"')


class MyDataset(BQDataset):
    dataset_id = 'rob'

    my_table = BQTable(
        name='my_table',
        appender_class=MyAppender,
        schema={
            'time': 'TIMESTAMP',
            'my_str': 'STRING',
            'my_int': 'INTEGER',
            'my_float': 'FLOAT',
        }

    )

    def push(self):
        time = datetime.datetime.now()
        for table in self.tables:
            table.append(time=time)


project_id = 'ambition-analytics'
ds = MyDataset(project_id, create_missing_dataset=True)
ds.push()
"""


class BQAppender(metaclass=abc.ABCMeta):
    """
    A BQTable instance needs to know how to grab data to append to the database.
    One of the arguments it takes is a appender class.  This is the base class you should
    derive from when creating your own Appenders.
    """

    @abc.abstractmethod
    def get_dataframe(self, **kwargs):
        """
        This method must be implemented.  It should contain the implementation of how
        you want to grab the data you want to push into the database.

        Any kwargs can be supplied to this method.  If you wish to validate the kwargs to this method,
        you can do so using the validate_kwargs(*kwargs) method.
        """

    def validate_kwargs(self, **kwargs):
        """
        Override this method to provide (optional) validation on the kwargs sent to the finalize_frame()
        method.
        """


class BQTable:
    """
    BQTable is a descriptor you should use when defining your BQDataset class.  It provides an
    interface for appending and accessing data from a particular table in your Bigquery Dataset.
    """

    # By default we use "DAY" parititoning for bigquery tables.
    _PARTITIONING_TYPE = "DAY"

    # This provides a lookup between pandas dtypes and bigquery database types.
    # It is used to validate schemas before appendin gdata
    _TYPE_LOOKUP = {
        pd.Timestamp: "TIMESTAMP",
        np.int64: "INTEGER",
        np.float64: "FLOAT",
        str: "STRING",
    }

    def __init__(
        self, name: str, schema: dict, appender_class: Optional[BQAppender] = None
    ):
        """
        Args:
            name: the name of the databse table to create/use in Bigquery
            schema: A dictionary with keys of field names and values of field types.
            appender_class: The class to use for appending data.
        """
        self.name = name
        self.schema = schema
        self.existance_confirmed = False
        self.appender_class = appender_class

    def __str__(self):
        return f"BQTable({self.name})"

    def __repr__(self):
        return self.__str__()

    def _table_exists(self, dataset):
        """
        Returns whether or not this table exists
        """
        return self.name in [
            t.table_id for t in dataset.client.list_tables(dataset.dataset_id)
        ]

    def _ensure_exists(self, dataset):
        """
        If it doesn't already exist, create this table on the provided dataset using the defined schema.
        """
        from google.cloud import bigquery

        if not self._table_exists(dataset):
            table_ref = dataset.dataset.table(self.name)
            schema = list(
                bigquery.SchemaField(field, dtype)
                for (field, dtype) in self.schema.items()
            )
            table = bigquery.Table(table_ref, schema=schema)
            table.partitioning_type = self._PARTITIONING_TYPE
            dataset.client.create_table(table)
        self.existance_confirmed = True

    def __get__(self, dataset, dataset_class):
        """
        This is what makes this class a descriptor
        """
        self._dataset = dataset
        self._ensure_exists(dataset)
        return self

    def _check_dataframe(self, df):
        """
        Checks to make sure a dataframe is valid for appending to the bigquery table
        """
        if df.empty:
            return

        expected_cols = set(self.schema.keys())
        received_cols = set(df.columns)
        missing_cols = expected_cols - received_cols
        if missing_cols:
            raise ValueError(f"Dataframe missing the following columns: {missing_cols}")

        bad_list = []
        for col, col_type in self.schema.items():
            mapped_type = self._TYPE_LOOKUP.get(type(df[col].iloc[0]))
            if col_type != mapped_type:
                bad_list.append(col)
        if bad_list:
            raise ValueError(f"Bad columns types for these columns: {bad_list}")

    def append_dataframe(self, df):
        """
        Call this method if you have a dataframe you want to manually append to the table.  Usually
        you won't want to use this method, but instead use the .append() method which will use your
        defined Appender class to create the dataframe and then append it to the bigquery table.
        """
        # Nothing to do if the dataframe is empty
        if df.empty:
            return

        # Make sure the dataframe looks right
        self._check_dataframe(df)

        # Make sure the frame is organized to be consistent with schema definition
        df = df[list(self.schema.keys())]

        # Push the dataframe to bigquery
        pandas_gbq.to_gbq(
            df,
            self.name_in_project,
            project_id=self._dataset.project_id,
            if_exists="append",
            progress_bar=False,
        )

    def get_dataframe(self, **kwargs):
        """
        This will pull in the source dataframe from this tables appender.  This is the dataframe
        that is used as the source for appending data to the bigquery table.
        """
        # Make sure that an appender class has been defined
        if self.appender_class is None:
            raise ValueError(
                "You must provide an appender class to the constructor if you want to call append()"
            )

        # Instantiate the appender class
        appender = self.appender_class()

        # Make sure the kwargs are valid
        appender.validate_kwargs(**kwargs)

        # Grab the dataframe
        df = appender.get_dataframe(**kwargs)

        return df

    def append(self, dry_run=False, **kwargs):
        """
        This is the preferred method for appending data to the bigquery database.  It will
        use the supplied Appender class to load a frame whose data will be pushed to bigquery.
        You can supply optional keyword arguments that will be passed to the .finalize_frame() method
        of your appender class.  These kwargs provide the ability to add whatever final/custom processing
        you want to your dataframe.
        """
        df = self.get_dataframe(**kwargs)

        # Append it to bigquery
        if not dry_run:
            self.append_dataframe(df)

    @ezr.cached_property
    def full_name(self):
        """
        This is the full name of the table.  It is what you should use in the
        FROM clause of any sql query.
        """
        return f"{self._dataset.project_id}.{self.name_in_project}"

    @ezr.cached_property
    def name_in_project(self):
        """
        The name of this table within the project
        """
        return f"{self._dataset.dataset_id}.{self.name}"

    @ezr.cached_property
    def latest_partition_time(self):
        """
        An attribute with a cached latest parition time
        """
        return self.get_latest_partition_time()

    @ezr.cached_container
    def latest_partition_frame(self):
        return self.get_latest_partition_frame()

    def get_latest_partition_frame(self):
        """
        An attribute with a cached dataframe of the latest parition data
        """
        sql = """
            SELECT
                *
            FROM
                {table}
            WHERE
                _PARTITIONDATE = "{partition_date_iso}"
        """.format(
            table=self.full_name,
            partition_date_iso=self.latest_partition_time.date().isoformat(),
        )
        df = self._dataset.query(sql)
        return df

    def get_latest_partition_time(self):
        """
        Grabs the most recent partition time for the table
        """
        sql_template = """
            SELECT
                _PARTITIONDATE AS partition_date
            FROM
                {full_name}
            WHERE
                _PARTITIONDATE >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
            ORDER BY
                partition_date DESC
            LIMIT 1
        """
        # This will exponentially increase the lookback time for partitions up to 256 days.
        days = 1
        while days <= 256:
            sql = sql_template.format(full_name=self.full_name, days=days)
            df = self._dataset.query(sql)
            if not df.empty:
                return df.partition_date.iloc[0].to_pydatetime()
            days = 2 * days
        raise RuntimeError(
            "To reduce data-size you can't query partitioned more than 256 days ago"
        )

    def get_latest_data(self, time_field_name):
        """
        If you have timestamped data in the database, this will grab all data with the latest timestamp
        for the latest partition.
        """

        sql = """
            WITH latest_partition AS (
                SELECT
                    *
                FROM
                    {table}
                WHERE
                    _PARTITIONDATE = "{partition_date_iso}"
            )
            SELECT
                *
            FROM
                latest_partition
            WHERE
                {time_field_name} = (SELECT MAX({time_field_name}) FROM latest_partition)
        """.format(
            table=self.full_name,
            time_field_name=time_field_name,
            partition_date_iso=self.latest_partition_time.date().isoformat(),
        )
        df = self._dataset.query(sql)
        return df


class BQDataset:
    def __init__(
        self, project_id, *, dataset_id=None, create_missing_dataset=False, **kwargs
    ):
        """
        A class that knows how to define and work with tables in a BigQuery dataset.

        Args:
            project_id: The project id this dataset belongs to
            dataset_id: The (optional) id for the dataset
            create_missing_dataset: Setting this flag will create non-existing datasets.
            kwargs: Any arguments you pass here will get passed to a .constructor_kwargs attribute.
        """
        # TODO: I don't like the interface I've chosen here.  I don't think a class variable is a good way
        # to go because it inhibits flexibility.  Also the way I chose project_id as an argument
        # and dataset_id as an optional argument is weird.  This needs improvement.

        if dataset_id is not None:
            self.dataset_id = dataset_id

        self.project_id = project_id
        self.constructor_kwargs = kwargs

        if not hasattr(self, "dataset_id"):
            raise ValueError(
                "You must set dataset_id as class-variable "
                "on your derived class or pass data_set_id to the constructor"
            )

        self._ensure_dataset_exists(create_missing_dataset)

    @ezr.cached_property
    def full_dataset_name(self):
        return f"{self.project_id}.{self.dataset_id}"

    @ezr.cached_property
    def all_dataset_names(self):
        return [d.dataset_id for d in self.client.list_datasets()]

    def _ensure_dataset_exists(self, create_missing_dataset):
        if self.dataset_id not in self.all_dataset_names:
            if create_missing_dataset:
                self._create_dataset()
            else:
                msg = (
                    f'Dataset "{self.dataset_id}" does not exist. '
                    "You can set create_missing_dataset=True to create it"
                )
                raise RuntimeError(msg)

    def ensure_all_tables_exist(self):
        for table in self.tables:
            table._ensure_exists(self)

    def _create_dataset(self):
        from google.cloud import bigquery

        dataset = bigquery.Dataset(self.full_dataset_name)
        dataset.location = "US"
        self.client.create_dataset(dataset, timeout=30)

    @property
    def auth_file(self):
        # TODO: Try to figure out how to init the contents of the auth file into env vars
        auth_file = None
        config_dir = os.path.expanduser("~/.config/gbq")
        if os.path.isdir(config_dir):
            auth_file_list = os.listdir(config_dir)
            if config_dir:
                auth_file = os.path.join(config_dir, auth_file_list[0])
        return auth_file

    @classmethod
    def get_table_names(cls):
        """
        A method to return all table names
        """
        table_names = []
        for name, obj in vars(cls).items():
            if isinstance(obj, BQTable):
                table_names.append(name)
        return table_names

    @ezr.cached_property
    def tables(self):
        """
        A list attribute containing all tables
        """
        tables = []
        for name in self.get_table_names():
            tables.append(getattr(self, name))
        return tables

    @property
    def on_dev_machine(self):
        """
        This will probably change when I do this for real, but for now, I check if I'm
        in a cloud function or not by checking for the existence of an auth file
        """
        return self.auth_file is not None

    @ezr.cached_property
    def client(self):
        """
        An attribute holding the google client
        """
        from google.cloud import bigquery
        from google.oauth2 import service_account

        client_kwargs = {"project": self.project_id}
        if self.on_dev_machine:
            client_kwargs.update(
                credentials=service_account.Credentials.from_service_account_file(
                    self.auth_file
                )
            )
        client = bigquery.Client(**client_kwargs)
        return client

    @ezr.cached_property
    def dataset(self):
        """
        An attribute holding the google dataset
        """
        return self.client.get_dataset(self.full_dataset_name)

    def query(self, sql_query, show_progress=False):
        """
        Run BigTable queries on this dataset. Tables can be referenced in FROM and JOIN SQL clauses
        using their .full_name attributes injected into a query template.
        """
        if show_progress:
            progress_bar_type = "tqdm"
        else:
            progress_bar_type = None

        df = pandas_gbq.read_gbq(
            sql_query,
            project_id=self.project_id,
            progress_bar_type=progress_bar_type,
        )
        return df
