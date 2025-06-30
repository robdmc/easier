import marimo

__generated_with = "0.12.10"
app = marimo.App(width="full", sql_output="pandas")


@app.cell
def _():
    # import easier.tools as ezr
    return


@app.cell
def _():
    import easier as ezr
    return (ezr,)


@app.cell
def _():
    from easier import tools as ezzr
    return (ezzr,)


@app.cell
def _(ezzr):
    ezzr
    return


@app.cell
def _(ezr):
    help(ezr.lomb_scargle)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
