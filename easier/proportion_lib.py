from functools import partial
from scipy import integrate
from scipy import stats
from textwrap import dedent
import holoviews as hv
import numpy as np

from functools import partial
from textwrap import dedent
from scipy import integrate
from scipy import stats
import numpy as np

class Proportion:
    """
    A class for analyzing proportions.

    ***************************************************
    print(ezr.Proportion.example)
    ***************************************************

    ...
    """
    example = dedent('\n        import easier as ezr\n        # Create a "small statistics" proportion with win = 60%\n        ps = ezr.Proportion(3, 5, name=\'small\')\n\n        # Create a "large statistics proportion\n        # that is "worse" than the small one with win = 50%\n        pb = ezr.Proportion(500, 1000, name=\'big\')\n\n        # We can set how to compare the distibutions\n        # Setting this to None (the default) will use\n        # the distribution means for comparison.  Setting\n        # it to some value in range (0, 1) will compare\n        # that percentile.\n        ps.set_default_global_comparison_quantile(.05)\n\n        # Print how much better the mean of one proportion is than the other\n        ps.set_default_global_comparison_quantile(None)\n        print(f\'Big is {pb - ps:0.2f} better than small based on mean\')\n\n        # Print how much better a quantile of one dist is than the other\n        ps.set_default_global_comparison_quantile(.05)\n        print(f\'Big is {pb - ps:0.2f} better than small based on q=.05\')\n\n        # All comparison operators are defined\n        print(\'big<small\', pb < ps)\n        print(\'big<=small\', pb <= ps)\n        print()\n        print(\'big>small\', pb > ps)\n        print(\'big>=small\', pb >= ps)\n        print()\n        print(\'big==small\', pb == ps)\n        print(sorted([ps, pb]))\n\n        # I can plot the distributions on top of each other\n        display(ps.plot() * pb.plot())\n\n        # I can plot the probability that one proportion is\n        # greater than the other\n        display(pb.plot_prob_better_curve(ps))\n    ')
    _DEFAULT_COMPARISON_QUANTILE = None

    def __init__(self, num_won, num_total, name='', plot_range=(0, 1), num_plot_points=1000):
        """
        Args:
            num_won: The number of events "won"
            num_total: The total number of events
            plot_range: The range of the plot
            num_plot_points: The number of points to plot
        """
        if num_won > num_total:
            raise ValueError('You cant win more than the total')
        if num_total < 1:
            raise ValueError('You must have at least one observation')
        self.num_won = num_won
        self.num_total = num_total
        self.alpha = num_won + 1
        self.beta = num_total - num_won + 1
        self.proportion = num_won / num_total
        self.dist = stats.beta(self.alpha, self.beta)
        self.name = name
        self.plot_range = plot_range
        self.num_plot_points = num_plot_points

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
            raise ValueError('Both proportion objects must have same comparsion_quantile')

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
        if self.name == '':
            prefix = ''
        else:
            prefix = f'{self.name}='
        return f'P({prefix}{self.num_won}/{self.num_total})'

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
        args = list(self.plot_range) + [self.num_plot_points]
        x = np.linspace(*args)
        c = hv.Curve((x, self.dist.pdf(x)), 'Proportion', 'Density', label=self.name)
        return c

    def plot_prob_better_curve(self, other):
        my_dist = self.dist
        their_dist = other.dist

        def integrand(delta, x):
            prob_me_at_x = my_dist.pdf(x)
            prob_at_least_delta_better = their_dist.cdf(x - delta)
            return prob_me_at_x * prob_at_least_delta_better

        def prob_at_least_better_than(delta):
            v_num = integrate.quad(partial(integrand, delta), 0, 1, points=[self.proportion, other.proportion])[0]
            return v_num
        my_min = self.dist.ppf(0.01)
        my_max = self.dist.ppf(0.99)
        their_min = other.dist.ppf(0.01)
        their_max = other.dist.ppf(0.99)
        minval = min([my_min, their_min])
        maxval = max([my_max, their_max])
        delta = maxval - minval
        if self.name == '':
            my_name = 'current distribution'
        else:
            my_name = self.name
        if other.name == '':
            their_name = 'other distribution'
        else:
            their_name = other.name
        xvals = np.linspace(-delta, delta, self.num_plot_points)
        yvals = [prob_at_least_better_than(x) for x in xvals]
        return hv.Curve((xvals, yvals), f'{my_name!r} is at least this much better than {their_name!r}', 'probability this is true')