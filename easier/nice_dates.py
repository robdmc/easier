# Directly copied from:
# https://gist.githubusercontent.com/zichongkao/d2b2082181dd9628add017ee71a8a64f/raw/799949af9d3c9c24c223149f9270250113f181e2/nice_dates.py


# import pandas as pd
# import matplotlib.dates as mdates
# from datetime import timedelta as tdelta
# from matplotlib.ticker import FuncFormatter
# import matplotlib.pylab as plt

INTERVALS = {
    "YEARLY": [1, 2, 4, 5, 10],
    "MONTHLY": [1, 2, 3, 4, 6],
    "DAILY": [1, 2],
    "WEEKLY": [1, 2],
    "HOURLY": [1, 2, 3, 4, 6, 12],
    "MINUTELY": [1, 5, 10, 15, 30],
    "SECONDLY": [1, 5, 10, 15, 30],
}

YAXIS_WIDTH_INCH = 2.0
TICKS_PER_INCH = 1.5


def _next_largest(value, options):
    for i in options:
        if i >= value:
            return i
    return i


def _get_dynamic_formatter(timedelta, *fmts):
    def dynamic_formatter(x, pos):
        import matplotlib.dates as mdates

        dx = mdates.num2date(x)
        strparts = [dx.strftime(fmt) for fmt in fmts]
        if pos > 0:
            # renders previous tick and removes common parts
            prior_dx = dx - timedelta
            prior_strparts = [prior_dx.strftime(fmt) for fmt in fmts]
            strparts = [
                new if new != prior else ""
                for new, prior in zip(strparts, prior_strparts)
            ]
        return "\n".join(strparts).strip()

    return dynamic_formatter


def _deduce_locators_formatters(max_ticks, data):
    from datetime import timedelta as tdelta
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter

    data_interval_seconds = (data[-1] - data[0]) / tdelta(seconds=1)
    interval_seconds = data_interval_seconds / max_ticks

    if interval_seconds < tdelta(minutes=0.5).total_seconds():
        # print("xticks: seconds")
        unit_multiple = _next_largest(interval_seconds, INTERVALS["SECONDLY"])
        timedelta = tdelta(seconds=unit_multiple)
        return (
            mdates.SecondLocator(bysecond=range(0, 60, unit_multiple)),
            FuncFormatter(_get_dynamic_formatter(timedelta, "%M%:S", "%-Hh", "%-d %b")),
        )
    elif interval_seconds < tdelta(hours=0.5).total_seconds():
        # print("xticks: minutes")
        unit_multiple = _next_largest(
            interval_seconds / tdelta(minutes=1).total_seconds(), INTERVALS["MINUTELY"]
        )
        timedelta = tdelta(minutes=unit_multiple)
        return (
            mdates.MinuteLocator(byminute=range(0, 60, unit_multiple)),
            FuncFormatter(_get_dynamic_formatter(timedelta, "%H%:M", "%-d %b", "%Y")),
        )
    elif interval_seconds < tdelta(days=0.5).total_seconds():
        # print("xticks: hours")
        unit_multiple = _next_largest(
            interval_seconds / tdelta(hours=1).total_seconds(), INTERVALS["HOURLY"]
        )
        timedelta = tdelta(hours=unit_multiple)
        return (
            mdates.HourLocator(byhour=range(0, 24, unit_multiple)),
            FuncFormatter(_get_dynamic_formatter(timedelta, "%-Hh", "%-d %b", "%Y")),
        )
    elif interval_seconds < tdelta(days=3).total_seconds():
        # print("xticks: days")
        unit_multiple = _next_largest(
            interval_seconds / tdelta(days=1).total_seconds(), INTERVALS["DAILY"]
        )
        timedelta = tdelta(days=unit_multiple)
        return (
            mdates.WeekdayLocator(byweekday=range(0, 7, unit_multiple)),
            FuncFormatter(_get_dynamic_formatter(timedelta, "%-d", "%b", "%Y")),
        )
    elif interval_seconds < tdelta(days=14).total_seconds():
        # print("xticks: weeks")
        unit_multiple = _next_largest(
            interval_seconds / tdelta(weeks=1).total_seconds(), INTERVALS["WEEKLY"]
        )
        timedelta = tdelta(days=unit_multiple * 7)
        return (
            mdates.WeekdayLocator(byweekday=0, interval=unit_multiple),
            FuncFormatter(_get_dynamic_formatter(timedelta, "%-d", "%b", "%Y")),
        )
    elif interval_seconds < tdelta(weeks=26).total_seconds():
        # print("xticks: months")
        unit_multiple = _next_largest(
            interval_seconds / tdelta(weeks=4).total_seconds(), INTERVALS["MONTHLY"]
        )
        timedelta = tdelta(weeks=unit_multiple * 4)
        return (
            mdates.MonthLocator(bymonth=range(1, 13, unit_multiple)),
            FuncFormatter(_get_dynamic_formatter(timedelta, "%b", "%Y")),
        )
    else:
        # print("xticks: years")
        unit_multiple = _next_largest(
            interval_seconds / tdelta(weeks=52).total_seconds(), INTERVALS["YEARLY"]
        )
        return (mdates.YearLocator(base=unit_multiple), mdates.DateFormatter("%Y"))


def nice_dates(ax=None):
    import matplotlib.pylab as plt
    import pandas as pd

    if ax is None:
        fig = plt.gcf()
        ax = plt.gca()
    else:
        fig = ax.figure

    # information for deciding tick locations
    xaxis_length_inch = fig.get_size_inches()[0] * fig.dpi / 100
    max_ticks = (xaxis_length_inch - YAXIS_WIDTH_INCH) * TICKS_PER_INCH
    data = pd.to_datetime(ax.lines[0].get_xdata())

    maj_locator, maj_formatter = _deduce_locators_formatters(max_ticks, data)

    ax.xaxis.set_major_locator(maj_locator)
    ax.xaxis.set_major_formatter(maj_formatter)
