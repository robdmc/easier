import datetime
import sys
import time
import traceback

from fleming import fleming
from dateutil.relativedelta import relativedelta

# inspired by
# https://schedule.readthedocs.io/en/stable/
# This was an early precursor to the crontabs project.
# I can probably remove it in favor of that



class Cron:
    """
    Usage:
    def my_task(name, msg=''):
        print(f'running at {datetime.datetime.now()}, name={name}, message={msg}')


    cron = Cron()

    cron.run(my_task, 'rob', msg='hello').every(seconds=4)

    """
    def __init__(self, robust=True):
        self._fleming_kwargs = {}
        self._relative_delta_kwargs = {}
        self._callable = None
        self._func_args = tuple()
        self._fun_kwargs = {}
        self._robust = robust

    def run(self, callable, *args, **kwargs):
        self._callable = callable
        self._func_args = args
        self._func_kwargs = kwargs
        self._start_if_possible()
        return self

    def every(self, **time_kwargs):
        for k, v in time_kwargs.items():
            if k.endswith('s'):
                self._relative_delta_kwargs[k] = v
                self._fleming_kwargs[k[:-1]] = v
            else:
                self._fleming_kwargs[k] = v
                self._relative_delta_kwargs[k + 's'] = v

        self._start_if_possible()
        return self

    def _start_if_possible(self):
        if self._callable is not None and self._fleming_kwargs:
            self._start()

    def _start(self):
        while True:
            try:
                now = datetime.datetime.now()
                previous_time = fleming.floor(now, **self._fleming_kwargs)
                next_time = previous_time + relativedelta(**self._relative_delta_kwargs)
                sleep_seconds = (next_time - now).total_seconds()
                time.sleep(sleep_seconds)
                self._callable(*self._func_args, **self._func_kwargs)
            except KeyboardInterrupt:
                sys.exit(0)
            except:  # noqa
                if self._robust:
                    print("v" * 60)
                    print("Exception in user code:")
                    traceback.print_exc(file=sys.stdout)
                    print("^" * 60)
                else:
                    raise
