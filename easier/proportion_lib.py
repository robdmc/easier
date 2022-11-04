from functools import partial
from scipy import integrate
from scipy import stats
import holoviews as hv
import numpy as np


class Proportion:
    def __init__(self, num_won, num_total, name=''):
        if num_won > num_total:
            raise ValueError('You cant win more than the total')
        if num_total < 1:
            raise ValueError('You must have at least one observation')
        self.alpha = num_won + 1
        self.beta = num_total - num_won + 1
        self.proportion = num_won / num_total
        self.dist = stats.beta(self.alpha, self.beta)
        self.name = name

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
        x = np.linspace(0, 1, 1600)
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

        my_min = self.dist.ppf(.01)
        my_max = self.dist.ppf(.99)

        their_min = other.dist.ppf(.01)
        their_max = other.dist.ppf(.99)

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

        xvals = np.linspace(-delta, delta, 100)
        yvals = [prob_at_least_better_than(x) for x in xvals]
        return hv.Curve((xvals, yvals), f'{my_name!r} is at least this much better than {their_name!r}', 'probability this is true')
