import numpy as np
from easier.bernstein import BernsteinFitter


def test_polynomial_coefficients_match_predict():
    # Generate some test data
    np.random.seed(42)
    x = np.linspace(-5, 5, 100)
    y = np.sin(x) + 0.1 * np.random.randn(len(x))

    # Fit the Bernstein polynomial
    fitter = BernsteinFitter()
    fitter.fit(x, y, degree=10)

    # Get polynomial coefficients
    coeffs = fitter.get_polynomial_coefficients()

    # Generate test points
    x_test = np.linspace(-5, 5, 50)

    # Get predictions using both methods
    y_bernstein = fitter.predict(x_test)
    y_poly = np.polyval(coeffs, x_test)

    # Check that they match within numerical tolerance
    np.testing.assert_allclose(y_bernstein, y_poly, rtol=1e-10, atol=1e-10)
