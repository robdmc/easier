class VonMisesFitter:
    def __init__(self, mod_value):
        """
        A fitter for periodic data using a von mises basis.
        Args:
            mod_value: This value mods to zero to make the data periodic
        """
        self.mod_value = mod_value
        self.model = None

    def _to_angle(self, x):
        """
        A utility to convert x to an angle between 0 and 2pi
        """
        import numpy as np

        x = np.array(x)
        return np.mod(x, self.mod_value) * 2 * np.pi / self.mod_value

    def fit(self, x, y, points=10, sigma_bin_factor=2, regularizer=None):
        """
        Fits the data to the von mises basis.
        Args:
            x: The x data
            y: The y data
            points: The number of points to use in computing the basis
            sigma_bin_factor: The standard deviation is the bin width times this factor
            regularizer: The regularizer to use in the fit.  If None, no regularizer is used
        """
        return self._fit(x, y, points, sigma_bin_factor, regularizer, predict=False)

    def fit_predict(self, x, y, points=10, sigma_bin_factor=2, regularizer=None):
        """
        Same as fit, but returns the prediction on the training x values
        """
        return self._fit(x, y, points, sigma_bin_factor, regularizer, predict=True)

    def _fit(
        self, x, y, points=10, sigma_bin_factor=2, regularizer=None, predict=False
    ):
        """
        A utility that does that actual fitting/predicting
        """
        import numpy as np
        from sklearn.linear_model import Ridge, LinearRegression

        self.points = points
        self.sigma_bin_factor = sigma_bin_factor

        x = np.array(x)
        y = np.array(y)
        X = self.get_design_matrix(x, points, sigma_bin_factor)

        if regularizer is None:
            self.model = LinearRegression(fit_intercept=False)
        else:
            self.model = Ridge(alpha=regularizer, fit_intercept=False)

        self.model.fit(X, y)
        if predict:
            return self.model.predict(X)

    def predict(self, x):
        """
        Predicts the y values for the given x values"""
        import numpy as np

        if self.model is None:
            raise ValueError("You must run .fit() before running .predict()")
        x = np.array(x)
        X = self.get_design_matrix(x, self.points, self.sigma_bin_factor)
        return self.model.predict(X)

    def get_design_matrix(self, x, points, sigma_bin_factor):
        """
        Given the x data and the number of points to use for constructing the basis, and the
        sigma_bin_factor, this function returns the design matrix.
        Args:
            x: The x data
            points: The number of points to use in computing the basis
            sigma_bin_factor: The standard deviation is the bin width times this factor
        """
        import numpy as np

        if points < 2:
            raise ValueError("Can only specifiy points > 1")

        from scipy.stats import vonmises

        x = np.array(x)
        theta = self._to_angle(x)

        points = int(points)
        points = np.linspace(0, 2 * np.pi, points + 1)[:points]

        delta = points[1] - points[0]
        sigma = sigma_bin_factor * delta
        kappa = 1 / (sigma**2)

        A = np.zeros((len(x), len(points)))
        for col, point in enumerate(points):
            dist = vonmises(kappa, loc=point)
            A[:, col] = dist.pdf(theta)
        return A
