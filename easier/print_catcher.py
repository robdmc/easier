import sys


class PrintCatcher(object):
    def __init__(self, stream="stdout"):
        self.text = ""
        if stream not in {"stdout", "stderr"}:
            raise ValueError('stream must be either "stdout" or "stderr"')
        self.stream = stream

    def write(self, text):
        self.text += text

    def flush(self):
        pass

    def __enter__(self):
        if self.stream == "stdout":
            sys.stdout = self
        else:
            sys.stderr = self
        return self

    def __exit__(self, *args):
        if self.stream == "stdout":
            sys.stdout = sys.__stdout__
        else:
            sys.stderr = sys.__stderr__
