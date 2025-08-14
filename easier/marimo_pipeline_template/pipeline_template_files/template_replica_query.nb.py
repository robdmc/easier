#! /usr/bin/env python3

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="full", sql_output="polars")

with app.setup:
    # Initialization code that runs before all other cells# Setup cell code
    import marimo as mo
    import numpy as np
    import pandas as pd
    import holoviews as hv
    import hvplot.pandas
    import os
    import sqlmodel
    import easier as ezr
    import inspect

    # from duckcacher import DuckCacher, duckloader_factory
    import polars as pl
    import inspect
    import textwrap
    import hvplot.pandas

    hv.extension("bokeh")
    hv_opts = {"width": 900, "height": 400, "tools": ["hover"]}
    hv.opts.defaults(
        hv.opts.Curve(**hv_opts),
        hv.opts.Scatter(**hv_opts),
        hv.opts.Histogram(**hv_opts),
        hv.opts.Bars(**hv_opts),
    )


@app.cell
def _():
    # Connection to replica db
    replica_db = sqlmodel.create_engine(os.environ["PGURL"])

    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    # Date ranges
    This is a template notebook.
    """
    )
    return


@app.cell
def _():
    class Loader(ezr.pickle_cache_mixin):
        def __init__(self, days_ago=90):
            self.days_ago = days_ago

        def run_query(self, sql):
            """Run a SQL query on the replica database."""

            pg = ezr.PG()
            df = pg.query(sql).to_dataframe()
            return df

        @ezr.cached_container
        def df(self):
            sql = inspect.cleandoc(
                f"""
                select
                    id
                from
                    hello_user
                limit 10
            """
            )
            return self.run_query(sql)

    loader = Loader()
    loader.df

    return (loader,)


@app.cell
def _(loader):
    cache = ezr.DuckCacher(file_name="stuff_from_replica.ddb")

    @cache.register
    def cases():
        return loader.df

    cache.sync()
    return


@app.cell
def _():
    ld = ezr.duckloader_factory("stuff_from_replica.ddb")
    ld.ls()
    return (ld,)


@app.cell
def _(ld):
    ld.df_cases
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
