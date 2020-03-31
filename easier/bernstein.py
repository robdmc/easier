import numpy as np
from scipy.special import comb
from scipy.interpolate import interp1d

class Compress:
    def __init__(self, is_sorted=True, min_val=None, max_val=None):
        self.is_sorted = is_sorted
        self.min_val = min_val
        self.max_val = max_val
        self.xmin = None
        self.ymin = None
        
    def set_limits(self, x):
        if not isinstance(x, np.ndarray):
            return

        if len(x) < 2:
            return

        if self.is_sorted:
            xmin = x[0]
            xmax = x[-1]
        else:
            xmin = np.min(x)
            xmax = np.max(x)
            
        if self.min_val is not None:
            xmin = self.min_val
        if self.max_val is not None:
            xmax = self.max_val

        self.xmin = xmin
        self.xmax = xmax
    
    def compress(self, x, learn_limits=True):
        """
        Compresses an sorted array to have min 0 and max 1
        """
        if isinstance(x, np.ndarray):
            x = x.flatten()
        if learn_limits:
            self.set_limits(x)
            
        if None in {self.xmin, self.xmax}:
            raise ValueError('Compressor never learned range limit')

        return (x - self.xmin) / (self.xmax - self.xmin)

    def expand(self, x):
        """
        Expands an array to go from [0, 1] interval to original interval
        """
        x = x.flatten()
        xmin = self.xmin
        xmax = self.xmax

        x = xmin + (xmax - xmin) * x
        return x


class Bern(Compress):
    EPS = 1e-15
    
    def __init__(self, func=None, x=None, y=None, N=500, is_sorted=True, min_val=None, max_val=None):
        super().__init__(is_sorted, min_val, max_val)
        self._validate_inputs(func, x, y, N)
        self.N = N
        if x is not None:
            self.x = self.compress(x)
        self.y = y
        
        if func is not None:
            self.func = lambda x: func(self.expand(x))
        else:
            self.func = self._get_interp_function()
            
    def _get_interp_function(self):
        return interp1d(self.x, self.y, fill_value=(self.y[0], self.y[-1]), bounds_error=False)
            
        
    def _validate_inputs(self, func, x, y, N):
        x_and_y_provided = (x is not None) and (y is not None)
        func_provided = func is not None
        
        if x_and_y_provided and func_provided:
            raise ValueError('You can only specify x-y pairs or a function.  Not both')
            
        if {x_and_y_provided, func_provided} != {True, False}:
            raise ValueError('You must specify either x-y pair or a function.')
            
        if N > 1000:
            raise ValueError(f'You entered N={N}.  Bad things can happen for N > 1000')
        
    def bern_term(self, n, k, x):
        """
        Returns the kth order term of a nth degree
        Bernstein polynomial
        """
        cc = comb(n, k)
        return cc * (1 + self.EPS - x) ** (n - k) * (x + self.EPS) ** k

    def get_fit_func(self, infunc):
        """
        Performs the appropriate berstein approximation
        expansion and sums up the terms
        """
        N = self.N
        k_vec = np.arange(N + 1)
        coeff_vec = infunc(k_vec / N)
        
        def bern_sum(x):
            if isinstance(x, np.ndarray):
                is_float = False
                x = x.flatten()
                X, K = np.meshgrid(x, k_vec)
                _, C = np.meshgrid(x, coeff_vec)
            else:
                is_float = True
                K = np.expand_dims(k_vec, -1)
                X = x * np.ones_like(K)
                C = np.expand_dims(coeff_vec, -1)
                
            B = self.bern_term(N, K, X)

            terms = C * B
            out = np.sum(terms, axis=0)
            if is_float:
                out = out[0]
            return out
        return bern_sum
        
    def get_fit_deriv(self, infunc):
        N = self.N
        k_vec = np.arange(N + 1)
        coeff_vec = infunc(k_vec / N)
        
        def bern_sum(x):
            if isinstance(x, np.ndarray):
                x = x.flatten()
                X, K = np.meshgrid(x, k_vec)
                _, C = np.meshgrid(x, coeff_vec)
            else:
                X = x
                K = k_vec
                C = x * coeff_vec
            
            
            B1 = self.bern_term(N-1, K-1, X)
            B2 = self.bern_term(N-1, K, X)

            terms = C *  N * (B1 - B2)
            out = np.sum(terms, axis=0)
            return out
        
        return bern_sum
    
    def predict(self, x):
        # learn_limits = not hasattr(self, 'xmax') or self.xmax is None
        # x = self.compress(x, learn_limits=learn_limits)
        func = self.get_fit_func(self.func)
        return func(self.compress(x))