from unittest import TestCase
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import easier as ezr

# # RUN TESTS WITH
# # pytest -sv ./test_ibis.py
# coverage erase && pytest -sv --cov=easier ./tests/test_ibis.py  | grep ibis
# coverage erase && pytest -sv --cov=easier.duck ./tests/test_duck.py::TestDuck::test_export_import


class TestOrderedSchema(TestCase):
    def setUp(self):
        # from ibis.expr import schema as sch

        df = pd.DataFrame({"my_int": np.array([1, 2, 3], dtype="int64")})
        df["my_float"] = 0.3 * df.my_int
        df["my_str"] = [np.NaN, "b", "c"]
        df["my_datetime"] = [
            pd.Timestamp("1/1/2000") + relativedelta(days=d, minutes=7, seconds=1.5)
            for d in range(3)
        ]
        df["unwanted"] = 7
        self.df = df

    def test_base_case(self):
        df = self.df.copy()
        dt = ezr.OrderedSchema.dt
        schema = ezr.OrderedSchema(
            {
                "my_float": dt.float32,
                "my_int": dt.int32,
                "my_datetime": dt.timestamp,
                "my_str": dt.string,
            }
        )
        df = schema.ordered_apply_to(df)

    def test_that_datetimes_cant_be_implicitely_changed_to_strings(self):
        df = self.df.copy()
        dt = ezr.OrderedSchema.dt
        schema = ezr.OrderedSchema(
            {
                "my_float": dt.float32,
                "my_int": dt.int32,
                "my_datetime": dt.string,
                "my_str": dt.string,
            }
        )
        with self.assertRaises(TypeError):
            df = schema.ordered_apply_to(df)

    def test_bad_float_conversion(self):
        df = self.df.copy()
        dt = ezr.OrderedSchema.dt
        schema = ezr.OrderedSchema(
            {
                "my_float": dt.float32,
                "my_int": dt.int32,
                "my_datetime": dt.string,
                "my_str": dt.float32,
            }
        )
        with self.assertRaises(TypeError):
            df = schema.ordered_apply_to(df)

    def test_bad_date_conversion(self):
        df = self.df.copy()
        dt = ezr.OrderedSchema.dt
        schema = ezr.OrderedSchema(
            {
                "my_float": dt.float32,
                "my_int": dt.int32,
                "my_datetime": dt.timestamp,
                "my_str": dt.timestamp,
            }
        )
        with self.assertRaises(TypeError):
            df = schema.ordered_apply_to(df)

    def test_non_strict_lets_things_slide(self):
        df = self.df.copy()
        dt = ezr.OrderedSchema.dt
        schema = ezr.OrderedSchema(
            {
                "my_float": dt.float32,
                "my_int": dt.int32,
                "my_datetime": dt.timestamp,
                "my_str": dt.timestamp,
            }
        )
        df = schema.ordered_apply_to(df, strict=False)

    def test_numerics_not_converted_to_strings(self):
        df = self.df.copy()
        dt = ezr.OrderedSchema.dt
        schema = ezr.OrderedSchema(
            {
                "my_float": dt.string,
                "my_int": dt.int,
                "my_datetime": dt.timestamp,
                "my_str": dt.string,
            }
        )
        with self.assertRaises(TypeError):
            df = schema.ordered_apply_to(df)

        df = self.df.copy()
        dt = ezr.OrderedSchema.dt
        schema = ezr.OrderedSchema(
            {
                "my_float": dt.float64,
                "my_int": dt.string,
                "my_datetime": dt.timestamp,
                "my_str": dt.string,
            }
        )
        with self.assertRaises(TypeError):
            df = schema.ordered_apply_to(df)

    def test_numerics_can_change_flavor(self):
        df = self.df.copy()
        dt = ezr.OrderedSchema.dt
        schema = ezr.OrderedSchema(
            {
                "my_float": dt.float32,
                "my_int": dt.int32,
                "my_datetime": dt.timestamp,
                "my_str": dt.string,
            }
        )
        df = schema.ordered_apply_to(df)

        dt = ezr.OrderedSchema.dt
        schema = ezr.OrderedSchema(
            {
                "my_float": dt.float64,
                "my_int": dt.int64,
                "my_datetime": dt.timestamp,
                "my_str": dt.string,
            }
        )
        df = schema.ordered_apply_to(df)

    def test_empty_okay(self):
        df = self.df.copy()
        df = df[df.unwanted == 12]
        dt = ezr.OrderedSchema.dt
        schema = ezr.OrderedSchema(
            {
                "my_float": dt.float32,
                "my_int": dt.int32,
                "my_datetime": dt.timestamp,
                "my_str": dt.string,
            }
        )
        df = schema.ordered_apply_to(df)
        self.assertEqual(len(df), 0)
