import io
import os
import pandas as pd
import numpy as np
import requests
import webbrowser
from .dataframe_tools import slugify as slugify_func
from typing import List, Optional


class SFDCEnv:
    def __init__(self):
        try:
            self.USERNAME = os.environ['SALESFORCE_USERNAME']
            self.PASSWORD = os.environ['SALESFORCE_PASSWORD']
            self.TOKEN = os.environ['SALESFORCE_SECURITY_TOKEN']
        except KeyError:
            raise KeyError(
                (
                    'Can\'t find salesforce environment variables.  Required variables are\n'
                    'SALESFORCE_USERNAME\n'
                    'SALESFORCE_PASSWORD\n'
                    'SALESFORCE_SECURITY_TOKEN\n'
                )
            )


class SalesForceReport(SFDCEnv):

    def __init__(self):
        super().__init__()

        # Import here to avoid simple_salesforce dependency at ezr import
        import simple_salesforce
        self.sf_obj = simple_salesforce.Salesforce(
            username=self.USERNAME, password=self.PASSWORD, security_token=self.TOKEN)

    @classmethod
    def open_report(cls, report_id):
        webbrowser.open_new(f'https://ambition.lightning.force.com/lightning/r/Report/{report_id}/view')

    def get_report(
            self,
            report_id: str,
            slugify: bool = False,
            date_fields: Optional[List[str]] = None,
            remove_copyright: bool = True
    ):
        url = f'https://{self.sf_obj.sf_instance}/{report_id}'
        self.req = requests.get(
            url,
            params=dict(export=1, xf='csv', enc='UTF-8', isdtp='p1'),
            cookies=dict(sid=self.sf_obj.session_id)
        )
        df = pd.read_csv(io.BytesIO(self.req.content))
        if slugify:
            df.columns = slugify_func(df.columns)

        if date_fields:
            for field in date_fields:
                df.loc[:, field] = df[field].astype(np.datetime64)

        if remove_copyright:
            df = df.iloc[:-5, :]
        return df


class Soql(SFDCEnv):
    """

    # Use soql to query sfdc
    soql = Soql(sf_obj=None)
    df = sfb.query('SELECT Id FROM Order')

    # Access the underlying simple-salesorce api object to manage objects
    soql.sf.Order.update(df.Id.iloc[0], {'Name': 'My new name'})

    # Reference
    # https://simple-salesforce.readthedocs.io/en/latest/user_guide/record_management.html

    """
    def __init__(self, sf_obj=None):
        if sf_obj is None:
            # Import here to avoid simple_salesforce dependency at ezr import
            import simple_salesforce
            self.sf = simple_salesforce.Salesforce(
                username=self.USERNAME,
                password=self.PASSWORD,
                security_token=self.TOKEN
            )
        else:
            self.sf = sf_obj

    def query(self, soql):
        rec_list = []
        for rec in self.sf.query(soql).get('records', []):
            rec.pop('attributes')
            rec_list.append(rec)

        df = pd.DataFrame(rec_list)
        return df
