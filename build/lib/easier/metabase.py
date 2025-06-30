import io
import os
import json


class Metabase:
    VALID_TYPES = ["category", "date", "datetime", "number", "text"]

    def __init__(self):
        # TODO: change the url to be read from the environment
        self.url_template = "https://metabase.app.blueberrypediatrics.com/api/card/{question_id}/query/csv?format_rows=true"
        self.headers = {
            "x-api-key": os.environ["METABASE_KEY"],
        }

    def _make_params_payload(self, **kwargs):
        parameters = {"parameters": []}

        for key, val in kwargs.items():
            try:
                param_type, param_value = val
            except:
                msg = "Kwargs error.  Must be like parameter_name=('cateory|date|datetime|number|text', parameter_value)"
                raise ValueError(msg)
            if param_type not in self.VALID_TYPES:
                raise ValueError(
                    f"Invalid type {param_type!r}.  Must be one of {self.VALID_TYPES}"
                )
            rec = {
                "type": param_type,
                "target": ["variable", ["template-tag", str(key)]],
                "value": [str(param_value)],
            }
            parameters["parameters"].append(rec)
        parameters["parameters"] = json.dumps(parameters["parameters"])
        return parameters

    def download_question(self, question_id, **kwargs):
        """
        Args:
            question_id: int = The question id you get from the metabase url
            kwargs = {'variable_name': ('cateory|date|datetime|number|text'), variable_value)}
        """
        import pandas as pd
        import requests

        url = self.url_template.format(question_id=question_id)
        params_dict = self._make_params_payload(**kwargs)
        if params_dict["parameters"]:
            post_kwargs = {"data": params_dict}
        else:
            post_kwargs = {}
        response = requests.post(url, headers=self.headers, **post_kwargs)
        if response.status_code != 200:
            raise ValueError(response.text)

        with io.StringIO(response.text) as buff:
            df = pd.read_csv(buff, low_memory=False)

        return df
