import io
import os


class Metabase:
    def __init__(self):
        self.url_template = "https://metabase-dot-parentapp-193923.ue.r.appspot.com/api/card/{question_id}/query/csv?format_rows=false"
        self.headers = {
            "x-api-key": os.environ["METABASE_KEY"],
            "Content-Type": "application/json",
        }

    def download_question(self, question_id):
        import pandas as pd
        import requests

        url = self.url_template.format(question_id=question_id)
        response = requests.post(url, headers=self.headers)
        if response.status_code != 200:
            raise ValueError(response.text)

        with io.StringIO(response.text) as buff:
            df = pd.read_csv(buff)

        return df
