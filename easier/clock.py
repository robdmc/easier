from collections import Counter
from contextlib import contextmanager
import datetime


class GlobalClock:
    delta = Counter()
    active_start_times = dict()

    def __init__(self):
        pass

    @contextmanager
    def running(self, *names):
        """
        Args:
            *names: *args of string names for clocks that should be running in the context
        """
        self.start(*names)
        yield
        self.stop(*names)

    @contextmanager
    def paused(self, *names):
        """
        Args:
            *names: *args of string names for clocks that should be paused in the context
        """
        allowed_names = set(self.active_start_times.keys())
        bad_names = set(names) - allowed_names
        if bad_names:
            raise ValueError(
                f"Trying to pause clocks {list(bad_names)} that aren't running."
            )
        if not names:
            raise ValueError("You must specify at least one clock name to pause")
        self.stop(*names)
        yield
        self.start(*names)

    def start(self, *names):
        """
        Args:
            *names: *args of string names for clocks that should be started
        """
        if not names:
            raise ValueError("You must provide at least one name to start")

        for name in names:
            if name not in self.active_start_times:
                self.active_start_times[name] = datetime.datetime.now()

    def stop(self, *names):
        """
        Args:
            *names: *args of string names for clocks that should be stopped

            If no names are provided, then stop all active clocks.
        """
        ending = datetime.datetime.now()
        if not names:
            names = list(self.active_start_times.keys())
        for name in names:
            if name in self.active_start_times:
                starting = self.active_start_times.pop(name)
                self.delta.update({name: (ending - starting).total_seconds()})

    def reset(self, *names):
        """
        Args:
            *names: *args of string names for clocks that should be reset

            If no names are provided, then stop and reset all clocks.
        """
        if not names:
            names = list(self.active_start_times.keys())
            names.extend(list(self.delta.keys()))
        for name in names:
            if name in self.delta:
                self.delta.pop(name)
            if name in self.active_start_times:
                self.active_start_times.pop(name)

    def get_time(self, *names):
        """
        Args:
            *names: *args of string names for clocks from which you want the current time
        Returns:
            If one name is provided, a floating point number with the time for that clock.
            If multiple names, a dict of {name: time} is returned for specified names
            If no names are provided, then a dict of all running clocks is returned
        """
        ending = datetime.datetime.now()
        if not names:
            names = list(self.delta.keys())
            names.extend(list(self.active_start_times.keys()))

        delta = Counter()
        for name in names:
            if name in self.delta:
                delta.update({name: self.delta[name]})
            elif name in self.active_start_times:
                delta.update(
                    {name: (ending - self.active_start_times[name]).total_seconds()}
                )
        if len(delta) == 1:
            return delta[list(delta.keys())[0]]
        else:
            return dict(delta)

    def __str__(self):
        """
        Print the current state of the clock.  Only include outputs for clocks that
        have been started and stopped.
        """
        records = sorted(self.delta.items(), key=lambda t: t[1], reverse=True)
        records = [("%0.6f" % r[1], r[0]) for r in records]

        if records:
            out_list = ["{: <15s}{}".format("seconds", "name")]
        else:
            out_list = []

        for rec in records:
            out_list.append("{: <15s}{}".format(*rec))

        return "\n".join(out_list)

    def __repr__(self):
        return self.__str__()


class Clock(GlobalClock):
    def __init__(self):
        # Override class attributes with instance attributes
        self.delta = Counter()
        self.active_start_times = dict()
