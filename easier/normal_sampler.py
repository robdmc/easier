from scipy import stats
import dataclasses
import easier as ezr
import holoviews as hv
import numpy as np
import typing

import dataclasses
import typing

@dataclasses.dataclass
class Sample:
    name: str
    mu: float
    sigma: float
    sigma_of_mean: float = dataclasses.field(init=False)
    n: int
    dist: typing.Any = dataclasses.field(init=False)
    dist_of_mean: typing.Any = dataclasses.field(init=False)

    def __post_init__(self):
        self.dist = stats.norm(loc=self.mu, scale=self.sigma)
        self.sigma_of_mean = self.sigma / np.sqrt(self.n)
        self.dist_of_mean = stats.norm(loc=self.mu, scale=self.sigma_of_mean)
        self.plot_min = self.mu - 5 * self.sigma
        self.plot_max = self.mu + 5 * self.sigma
        self.plot_min_of_mean = self.mu - 5 * self.sigma / np.sqrt(self.n)
        self.plot_max_of_mean = self.mu + 5 * self.sigma / np.sqrt(self.n)

    def plot(self, filled=True, use_uncertainty_of_mean=False):
        plot_points = 300
        if use_uncertainty_of_mean:
            dist = self.dist_of_mean
            xmin, xmax = (self.plot_min_of_mean, self.plot_max_of_mean)
        else:
            dist = self.dist
            xmin, xmax = (self.plot_min, self.plot_max)
        x = np.linspace(xmin, xmax, plot_points)
        y = dist.pdf(x)
        c = hv.Curve((x, y), label=self.name)
        if filled:
            c = hv.Area(c).options(alpha=0.15)
        return c

class NormalSampleJoiner:

    def __init__(self):
        self.samples = ezr.Item()
        self.samples.combined = None

    def _check_args(self, data, mu, sigma, n):
        if data is None:
            if None in {mu, sigma, n}:
                raise ValueError("You must specify mu, sigma and n when you don't supply data")
        elif {None} != {mu, sigma, n}:
            raise ValueError('You cannot specify mu ,sigma or n when you supply data')

    def __str__(self):
        if self.samples.combined is None:
            return 'combined ~ (mu=None, sigma=None, n=None)'
        else:
            return f'combined ~ (mu={self.samples.combined.mu:.2e}, sigma={self.samples.combined.sigma:.2e}, n={self.samples.combined.n:.2e})'

    def __repr__(self):
        return self.__str__()

    def _get_valid_name(self, name):
        if name == 'combined':
            raise ValueError("You cannot name a sample 'combined'.  That name is reserved.")
        if name is None:
            name = f'sample_{len(self.samples):03d}'
        return name

    def add_sample(self, *, data=None, mu=None, sigma=None, n=None, name=None):
        self._check_args(data, mu, sigma, n)
        name = self._get_valid_name(name)
        if data is not None:
            data = np.array(data)
            mu = np.mean(data)
            sigma = np.std(data)
            n = len(data)
        sample = Sample(name=name, mu=mu, sigma=sigma, n=n)
        self._ingest(name, sample)

    def _ingest(self, name, sample):
        self.samples[name] = sample
        if self.samples.combined is None:
            self.samples.combined = sample
        else:
            new_combined = self.combine_samples(self.samples.combined, sample)
            self.samples.combined = new_combined

    def combine_samples(self, sample_a, sample_b, name='combined'):
        mu_a = sample_a.mu
        sigma_a = sample_a.sigma
        n_a = sample_a.n
        mu_b = sample_b.mu
        sigma_b = sample_b.sigma
        n_b = sample_b.n
        mu = n_a * mu_a / (n_a + n_b) + n_b * mu_b / (n_a + n_b)
        term1_numer = n_a * (sigma_a ** 2 + mu_a ** 2) + n_b * (sigma_b ** 2 + mu_b ** 2)
        term1_denom = n_a + n_b
        term2_numer = (n_a * mu_a + n_b * mu_b) ** 2
        term2_denom = (n_a + n_b) ** 2
        sigma2 = term1_numer / term1_denom - term2_numer / term2_denom
        sigma = np.sqrt(sigma2)
        n = n_a + n_b
        return Sample(name=name, mu=mu, sigma=sigma, n=n)

    def plot(self, use_uncertainty_of_mean=False, legend_position='top', filled=True):
        c_list = []
        for sample in self.samples.values():
            c_list.append(sample.plot(filled=filled, use_uncertainty_of_mean=use_uncertainty_of_mean))
        return hv.Overlay(c_list).options(legend_position=legend_position)