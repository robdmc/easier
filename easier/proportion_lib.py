from functools import partial
from textwrap import dedent
from scipy import integrate
from scipy import stats
import numpy as np


class Proportion:
    """
    A class for analyzing proportions.

    print(ezr.Proportion.example)

    ...
    """

    example = dedent(
        """
        import easier as ezr
        # Create a "small statistics" proportion with win = 60%
        ps = ezr.Proportion(3, 5, name='small')

        # Create a "large statistics proportion
        # that is "worse" than the small one with win = 50%
        pb = ezr.Proportion(500, 1000, name='big')

        # We can set how to compare the distibutions
        # Setting this to None (the default) will use
        # the distribution means for comparison.  Setting
        # it to some value in range (0, 1) will compare
        # that percentile.
        ps.set_default_global_comparison_quantile(.05)

        # Print how much better the mean of one proportion is than the other
        ps.set_default_global_comparison_quantile(None)
        print(f'Big is {pb - ps:0.2f} better than small based on mean')

        # Print how much better a quantile of one dist is than the other
        ps.set_default_global_comparison_quantile(.05)
        print(f'Big is {pb - ps:0.2f} better than small based on q=.05')

        # All comparison operators are defined
        print('big<small', pb < ps)
        print('big<=small', pb <= ps)
        print()
        print('big>small', pb > ps)
        print('big>=small', pb >= ps)
        print()
        print('big==small', pb == ps)
        print(sorted([ps, pb]))

        # I can plot the distributions on top of each other
        display(ps.plot() * pb.plot())

        # I can plot the probability that one proportion is
        # greater than the other
        display(pb.plot_prob_better_curve(ps))
    """
    )

    _DEFAULT_COMPARISON_QUANTILE = None

    def __init__(self, num_won, num_total, name=""):
        """
        Args:
            num_won: The number of events "won"
            num_total: The total number of events
        """
        if num_won > num_total:
            raise ValueError("You cant win more than the total")
        if num_total < 1:
            raise ValueError("You must have at least one observation")
        self.num_won = num_won
        self.num_total = num_total
        self.alpha = num_won + 1
        self.beta = num_total - num_won + 1
        self.proportion = num_won / num_total
        self.dist = stats.beta(self.alpha, self.beta)
        self.name = name

    @property
    def comparison_quantile(self):
        return self._DEFAULT_COMPARISON_QUANTILE

    @property
    def nominal_value(self):
        if self.comparison_quantile is None:
            return self.dist.mean()
        else:
            return self.dist.ppf(self.comparison_quantile)

    def set_default_global_comparison_quantile(self, value):
        self.__class__._DEFAULT_COMPARISON_QUANTILE = value

    def _ensure_comparable(self, other):
        if len({self.comparison_quantile, other.comparison_quantile}) > 1:
            raise ValueError(
                "Both proportion objects must have same comparsion_quantile"
            )

    def __sub__(self, other):
        self._ensure_comparable(other)
        return self.nominal_value - other.nominal_value

    def __eq__(self, other):
        self._ensure_comparable(other)
        return self.nominal_value == other.nominal_value

    def __lt__(self, other):
        self._ensure_comparable(other)
        return self.nominal_value < other.nominal_value

    def __le__(self, other):
        self._ensure_comparable(other)
        return self.nominal_value <= other.nominal_value

    def __gt__(self, other):
        self._ensure_comparable(other)
        return self.nominal_value > other.nominal_value

    def __ge__(self, other):
        self._ensure_comparable(other)
        return self.nominal_value >= other.nominal_value

    def __str__(self):
        if self.name == "":
            prefix = ""
        else:
            prefix = f"{self.name}="
        return f"P({prefix}{self.num_won}/{self.num_total})"

    def __repr__(self):
        return self.__str__()

    def probability_better_than(self, other, amount=0):
        my_dist = self.dist
        their_dist = other.dist

        def prob_better(x):
            prob_me_at_x = my_dist.pdf(x)
            prob_they_are_less = their_dist.cdf(x - amount)
            prob_me_at_x_and_better_than_them = prob_me_at_x * prob_they_are_less
            return prob_me_at_x_and_better_than_them

        v_num = integrate.quad(prob_better, 0, 1, points=[self.proportion])[0]
        return v_num

    def plot(self):
        import holoviews as hv

        x = np.linspace(0, 1, 1600)
        c = hv.Curve((x, self.dist.pdf(x)), "Proportion", "Density", label=self.name)
        return c

    def plot_prob_better_curve(self, other):
        import holoviews as hv

        my_dist = self.dist
        their_dist = other.dist

        def integrand(delta, x):
            prob_me_at_x = my_dist.pdf(x)
            prob_at_least_delta_better = their_dist.cdf(x - delta)
            return prob_me_at_x * prob_at_least_delta_better

        def prob_at_least_better_than(delta):
            v_num = integrate.quad(
                partial(integrand, delta),
                0,
                1,
                points=[self.proportion, other.proportion],
            )[0]
            return v_num

        my_min = self.dist.ppf(0.01)
        my_max = self.dist.ppf(0.99)

        their_min = other.dist.ppf(0.01)
        their_max = other.dist.ppf(0.99)

        minval = min([my_min, their_min])
        maxval = max([my_max, their_max])
        delta = maxval - minval

        if self.name == "":
            my_name = "current distribution"
        else:
            my_name = self.name

        if other.name == "":
            their_name = "other distribution"
        else:
            their_name = other.name

        xvals = np.linspace(-delta, delta, 100)
        yvals = [prob_at_least_better_than(x) for x in xvals]
        return hv.Curve(
            (xvals, yvals),
            f"{my_name!r} is at least this much better than {their_name!r}",
            "probability this is true",
        )
