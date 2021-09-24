class Elliptic:
    def __init__(self, kind, f_pass, f_stop, max_suppression_pass, min_suppression_stop, f_sample=1):
        """
        Sets up a class for digitally filtering time series signals.
        Args:
            kind: selection from ['lowpass', 'highpass', 'bandpass', 'bandstop']

            f_pass: The frequency or frequencies at which the passband starts to roll off
                    For low/high pass, this is a scalar.  For bandpass/stop this is a pair of [low_freq, high_freq]

            f_stop: The frequency or frequencies at which the stopband is at full strength
                    For low/high pass, this is a scalar.  For bandpass/stop this is a pair of [low_freq, high_freq]

            max_suppression_pass: Don't suppress signals in the passband by more than this many dB

            min_suppression_stop: Make sure entire passband is suppressed by at least this many dB

            f_sample: This is the sample frequency to use

        """
        self._kind = kind
        self._f_pass = f_pass
        self._f_stop = f_stop
        self._f_sample = f_sample
        self._max_suppression_pass = max_suppression_pass
        self._min_suppression_stop = min_suppression_stop

        self._check_kind(kind)
        self._check_freqs(f_pass, f_stop, f_sample)

    def _check_kind(self, kind):
        allowed_kinds = ['lowpass', 'highpass', 'bandpass', 'bandstop']
        if kind not in allowed_kinds:
            raise ValueError(f'kind must be taken from {allowed_kinds}')

    def _check_freqs(self, f_pass, f_stop, f_sample):  # noqa
        if self._kind == 'lowpass':
            if not (f_pass < f_stop):
                raise ValueError('You must make sure f_pass < f_stop')

        elif self._kind == 'highpass':
            if not (f_stop < f_pass):
                raise ValueError('You must make sure f_stop < f_pass')

        elif self._kind == 'bandpass':
            elements_okay = (f_stop[0] < f_stop[1]) and (f_pass[0] < f_pass[1])
            left_okay = (f_stop[0] < f_pass[0])
            right_okay = f_pass[1] < f_stop[1]

            if not (elements_okay and left_okay and right_okay):
                raise ValueError('The ordering of your band frequencies is incorrect')

        elif self._kind == 'bandstop':
            elements_okay = (f_stop[0] < f_stop[1]) and (f_pass[0] < f_pass[1])
            left_okay = (f_pass[0] < f_stop[0])
            right_okay = f_stop[1] < f_pass[1]

            if not (elements_okay and left_okay and right_okay):
                raise ValueError('The ordering of your band frequencies is incorrect')

        if self._kind in ['bandpass', 'bandstop']:
            freqs = list(f_pass) + list(f_stop)
        else:
            freqs = [f_pass, f_stop]

        if max(freqs) > (2 * f_sample):
            raise ValueError('Your filter frequencies must be less than half your sampling frequency')

    def _get_filter_coeffs(self, output='sos', analog=False):
        from scipy import signal

        # Compute the filter params
        N, Wn = signal.ellipord(
            wp=self._f_pass,
            ws=self._f_stop,
            gpass=self._max_suppression_pass,
            gstop=self._min_suppression_stop,
            analog=False,
            fs=self._f_sample
        )
        if analog:
            fs = None
        else:
            fs = self._f_sample

        coeffs = signal.ellip(
            N,
            rp=self._max_suppression_pass,
            rs=self._min_suppression_stop,
            Wn=Wn,
            btype=self._kind,
            analog=analog,
            output=output,
            fs=fs
        )

        return coeffs

    def filter(self, y, symmetric=False):
        """
        Run the specified filter over data
        Args:
            y: The data to filter
            symmetric: If set to True, the filtfilt scipy function will be used to do symmetric filtering

        Returns:
            The filtered time series
        """
        from scipy import signal

        # Get the filter coeffs
        sos = self._get_filter_coeffs()

        # Run the filter
        if symmetric:
            yf = signal.sosfiltfilt(sos, y, )
        else:
            yf = signal.sosfilt(sos, y)

        return yf

    def plot_response(self, n_points=5_000):
        """
        Plot the theoretical response of the filter.
        Args:
            n_points: The number of points to draw in the response curve

        Returns:
            A holoviews plot of the theoretical response
        """
        from scipy import signal
        import holoviews as hv
        import numpy as np
        sos = self._get_filter_coeffs()
        w, h = signal.sosfreqz(sos, worN=n_points, fs=self._f_sample)

        c = hv.Curve((w[1:], 20 * np.log10(abs(h[1:]) + 1e-100)), 'Frequency', 'Response (dB)')
        return c

    def plot_noise_response(self, n_points=5_000):
        """
        Plot the frequency response of a random unit normal timeseries
        Args:
            n_points: The number of points to draw in the response curve

        Returns:
            A holoviews plot of the filted noise suppression
        """
        import numpy as np
        from scipy import signal
        import holoviews as hv
        y = np.random.randn(n_points)
        y = y - np.mean(y)
        yf = self.filter(y)

        freq, pwr = signal.periodogram(y, fs=self._f_sample)
        freq_f, pwr_f = signal.periodogram(yf, fs=self._f_sample)

        pwr_db = 10 * np.log10(pwr + 1e-100)
        pwr_f_db = 10 * np.log10(pwr_f + 1e-100)

        pwr_diff = pwr_f_db - pwr_db

        return hv.Curve((freq[1:], pwr_diff[1:]), 'Frequency', 'Response (dB)')
