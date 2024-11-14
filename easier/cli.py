import sys
import click
import subprocess
import os


from easier.gsheet import Email, CONFIG_FILE_NAME

try:
    email = Email(CONFIG_FILE_NAME).email
except:  # noqa
    pass


@click.group()
def cli():
    pass


@click.command(help="Read a salesforce report as csv and send results to stdout")
@click.argument("report_id")
def sfdc(report_id):
    from easier.salesforce import SalesForceReport

    sfdc = SalesForceReport()
    df = sfdc.get_report(report_id)
    df.to_csv(sys.stdout, index=False)


@click.command(
    help=f"Read a google sheet as csv and send results to stdout.\nShare document with email: {email}"
)
@click.argument("document_name")
@click.argument("tab_name")
def gsheet(document_name, tab_name):
    from easier.gsheet import GSheet

    gsheet = GSheet(document_name, tab_name)
    df = gsheet.to_dataframe_as_values()
    df.to_csv(sys.stdout, index=False)


@click.command(
    help=f"Read csv from stdin and push to coordinates on sheet.\nShare document with email: {email}"
)
@click.argument("document_name")
@click.argument("tab_name")
@click.argument("upper_left")
@click.option(
    "-c", "--clear-to-bottom", is_flag=True, help="Empty all cells below the table"
)
def gsheet_push(document_name, tab_name, upper_left, clear_to_bottom):
    import pandas as pd
    from easier.gsheet import GSheet

    df = pd.read_csv(sys.stdin)
    gsheet = GSheet(document_name, tab_name)
    gsheet.store_frame_to_coords(df, "A3", clear_to_bottom=clear_to_bottom)


@click.command(help="Run all cells in a notebook until the first debug setpoint is hit")
@click.argument("file_name")
def nb_debug(file_name):
    file_name = os.path.realpath(os.path.expanduser(file_name))
    script_name = file_name.replace(".ipynb", ".py")
    if os.path.exists(script_name):
        os.remove(script_name)
    cmd = "jupyter nbconvert --to script " + file_name
    subprocess.check_output(cmd, shell=True)
    os.execlp("python", "python", script_name)
