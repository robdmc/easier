#! /usr/bin/env python

from duck import Table, duck_connection, Duck
import os
import pandas as pd
import numpy as np

db_file = "/tmp/deleteme.ddb"
if os.path.isfile(db_file):
    os.unlink(db_file)

df1 = pd.DataFrame({"a": [1, 2, 3], "b": [5, 6, 7]})
df2 = pd.DataFrame({"c": [1, 2, 3], "d": [5, 6, 7]})

duck = Duck(db_file)

duck.tables.create("one", df1)


sql = """
    select
        *
    from
        one
    cross join
        two
"""
dfo = duck.query(sql, two=df2)
print(dfo.to_string())
