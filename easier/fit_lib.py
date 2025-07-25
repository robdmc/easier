from textwrap import dedent
import easier as ezr


class examples:
    """
    A descriptor whose only purpose is to print help text
    """

    def __get__(self, *args, **kwargs):
        print(
            dedent(
                "\n            from easier import Fit\n\n            #=================================================================\n            #                Simple Model Fitting:  y ~ model(x)\n            #=================================================================\n            # Make data from noise-corrupted sinusoid\n            x = np.linspace(0, 2 * np.pi, 100)\n            y = np.sin(x) - .7 * np.cos(x) + .1 * np.random.randn(len(x))\n\n            # Make up some artificial weights to apply to fit\n            w = np.exp(8 * y)\n\n\n            # Define a model function you want to fit to\n            # All model parameters are on the p object.\n            # The names \"x\", and \"y\" are reserved for the data you are fitting\n            def model(p):\n                return p.a * np.sin(p.k * p.x) + p.b * np.cos(p.k * p.x)\n\n            # Initialize a fitter with purposefully bad guesses\n            fitter = ezr.Fitter(a=-1, b=2, k=.2)\n\n            # Fit the data with weights and plot fit quality every 5 iterations\n            fitter.fit(x=x, y=y, weights=w, model=model, plot_every=5)\n\n            # Plot the final results\n            display(fitter.plot())\n            display(fitter.params.df)\n\n            #=================================================================\n            #                Advanced Fitting:  y ~ model(x)\n            #=================================================================\n            # Make data from noise-corrupted sinusoid\n            x = np.linspace(0, 2 * np.pi, 100)\n            y = np.sin(x) - .7 * np.cos(x) + .1 * np.random.randn(len(x))\n\n            # Define a model function you want to fit to\n            # All model parameters are on the p object.\n            # The names \"x\", and \"y\" are reserved for the data you are fitting\n            def model(p):\n                return p.a * np.sin(p.k * p.x) + p.b * np.cos(p.k * p.x)\n\n            # Initialize a fitter with purposefully bad guesses\n            fitter = ezr.Fitter(a=-1, b=2, k=.2)\n\n            # Fit the data and plot fit quality every 5 iterations\n            fitter.fit(\n                x=x,                   # The independent data\n                y=y,                   # The dependent data\n                model=model,           # The model function\n                plot_every=5,          # Plot fit every this number of iterations\n                algorithm='fmin_bfgs', # Scipy optimization routine to use\n                verbose=False          # Don't print convergence info\n            )\n\n            # Get predictions at specific values\n            x_predict = np.linspace(0, 6 * np.pi, 300)\n            y_predict = fitter.predict(x_predict)\n\n            # Get the components of the fit chart\n            components = fitter.plot(\n                x=x_predict,\n                scale_factor=10,\n                label='10X Scaled Fit',\n                line_color='red',\n                scatter_color='blue',\n                size=15,\n                xlabel='My X Label',\n                ylabel='My Y Label',\n                as_components=True,\n            )\n\n            # Display the components as a layout rather than overlay\n            display(hv.Layout(components))\n\n            #=================================================================\n            #                Advanced Fitting:  minimize: cost(x, y, p)\n            #=================================================================\n            # Assume we have x, y that are a third order polymial\n            x_data = np.linspace(0, 1, 100)\n            y_data = .7 * x_data ** 3 + .2 * x_data ** 2 + .1 * x_data\n            y_data = y_data + .01 * np.random.randn(len(y_data))\n\n            # Say that we know for sure that the function polyomial we want\n            # has a zero intercept.  Furthermore, say we know for sure that\n            # the sum of the coefficients have to equal to 1.  We construct\n            # a cost function to encode this.\n            def cost(p):\n                # A full polynomial fit with all orders (so they can be thrown into polyfit)\n                yf = p.a * p.x ** 3 + p.b * p.x ** 2 + p.c * p.x + p.d\n\n                # Compute the sum of squares error cost of the fit from the data\n                fit_cost = np.sum((yf - p.y)**2)\n\n                # Regularize the length of the parameters to be one, and the intercept term to be zero.\n                param_cost = 100 * len(p.x) * (p.a + p.b + p.c - 1) ** 2 + 10000 * len(p.x) * p.d ** 2\n\n                # Return the total cost\n                return fit_cost + param_cost\n\n            # Create a fitter by initializing the the params to starting positions\n            f = ezr.Fitter(a=1/3., b=1/3., c=1/3., d=0.)\n\n            # Run the fit\n            f.fit(x=x_data, y=y_data, cost=cost)\n            p = f.params\n\n            # Print the results\n            print(f.params)\n            "
            )
        )
        return None


class Fitter:
    """
    This class is a convenience wrapper for scipy optimization.
    Look at Fitter.examples attribute to see usage.

    The constructor takes exactly the same arguments as ParamState.
    """

    OPTIMIZER_NAMES = {"fmin", "fmin_powell", "fmin_cg", "fmin_bfgs"}
    DEFAULT_VERBOSE = True
    examples = examples()

    def __init__(self, *args, **kwargs):
        """
        *args and **kwargs define/initialize the model variables.
        They are passed directly to a ParamState constructor
        """
        from .param_state import ParamState  # type: ignore

        self._params = ParamState(*args, **kwargs)
        self._algorithm = "fmin"
        self._optimizer_kwargs = {}
        self._model = None
        self._givens = {}
        self._cost = self._default_cost
        self._verbose = self.DEFAULT_VERBOSE

    @property
    def all_params(self):
        """
        Show all parameters (even data variables)
        """
        p = self._params.clone()
        return p

    @property
    def params(self):
        """
        Show all params except data variables
        """
        p = self.all_params
        p.drop("x")
        p.drop("y")
        return p

    def extra(self, **kwargs):
        """
        Put extra attributes on the params object.  These params will
        be completely ignored by the optimizer.  This is good for passing
        utility functions to your model or cost fuctions.
        """
        for key, val in kwargs.items():
            setattr(self._params, key, val)

    def given(self, **kwargs):
        """
        Supply kwargs that will set constant variables.  These can either
        be new variables or already defined variables.  This allows you
        to easily turn on and off variables to optimize.
        """
        self._givens = kwargs
        return self

    def optimizer_kwargs(self, **kwargs):
        """
        Additional kwargs to pass to optimizer.
        """
        self._optimizer_kwargs = kwargs
        return self

    def _cost_wrapper(self, args, p):
        """
        Allows users to define their cost functions as only functions
        of p, the ParamState object.
        """
        p.ingest(args)
        return self._cost(p)

    def _default_cost(self, p):
        """
        Allows users to define their cost functions as only functions
        of p, the ParamState object.
        """
        import numpy as np

        if self._model is None:
            raise ValueError("No model defined.")
        if hasattr(p, "weights"):
            w = p.weights
        else:
            w = np.ones_like(p.y)
        z = (self._model(p) - p.y) * w / np.sum(w)
        return np.sum(z**2)

    def _plotter(self, *args, **kwargs):
        """
        A function to plot fits in real time
        """
        import holoviews as hv

        if kwargs["data"]:
            xd, yd = kwargs["data"][-2:]
        else:
            xd, yd = ([0], [0])
        fit_line = hv.Curve(*args, **kwargs)
        data = hv.Scatter((xd, yd))
        overlay = hv.Overlay([fit_line, data])
        overlay = overlay.opts(hv.opts.Curve(color=ezr.cc.b), hv.opts.Scatter(color=ezr.cc.a, size=5, alpha=0.5))  # type: ignore
        return overlay

    def _model_wrapper(self, model):
        """
        The model wrapper will wrap the model with a function
        that updates real time plots of fit if needed.
        """
        if model is None:
            return model
        if self.plot_every is None:
            return model
        else:
            self.plot_counter = 0

            def wrapped(*args, **kwargs):
                yfit = model(*args, **kwargs)
                if self.plot_counter % self.plot_every == 0:  # type: ignore
                    self.pipe.send((self._params.x, yfit, self._params.x, self._params.y))  # type: ignore
                self.plot_counter += 1
                return yfit

            return wrapped

    def fit(
        self, *, x=None, y=None, weights=None, model=None, cost=None, plot_every=None, algorithm="fmin", verbose=True
    ):
        """
        The method to use for training the fitter.
        Args:
                     x: the indepenant data
                     y: the depenant data
                 model: a model function to fit the data to
                  cost: a custom cost function (defaults to least squares)
            plot_every: Plot solution in real time every this number of iterations
             algorithm: Scipy optimization routine to use.  Enter nonsense to see list of valid.
               verbose: Print convergence information
        """
        from scipy import optimize
        from IPython.display import display
        from holoviews.streams import Pipe
        import holoviews as hv
        import numpy as np

        self.plot_every = plot_every
        if algorithm not in self.OPTIMIZER_NAMES:
            raise ValueError(f"Invalid optimizer {algorithm}.  Choose one of {self.OPTIMIZER_NAMES}")
        givens = dict(x=x, y=y)
        if weights is not None:
            givens.update({"weights": weights})
        givens.update(self._givens)
        self._params.given(**givens)
        if plot_every is not None:
            x, y = (self._params.x, self._params.y)  # type: ignore
            xmin, xmax = (np.min(x), np.max(x))
            ymin, ymax = (np.min(y), np.max(y))
            scale = 0.1
            delta_x = scale * (xmax - xmin)
            delta_y = scale * (ymax - ymin)
            xmin, xmax = (xmin - delta_x, xmax + delta_x)
            ymin, ymax = (ymin - delta_y, ymax + delta_y)
            xlim = (xmin, xmax)
            ylim = (ymin, ymax)
            self.pipe = Pipe(data=[])
            try:
                dmap = hv.DynamicMap(self._plotter, streams=[self.pipe])
                dmap.opts(hv.opts.Overlay(xlim=xlim, ylim=ylim))
                display(dmap)
            except AttributeError:
                raise RuntimeError("You must import holoviews and set bokeh backround for plotting to work")
        if model is None and cost is None:
            raise ValueError("You must supply either a model function or a cost function")
        self._raw_model = model
        self._model = self._model_wrapper(self._raw_model)
        a0 = self._params.array
        if cost is not None:
            self._cost = cost
        optimizer = getattr(optimize, algorithm)
        self._optimizer_kwargs.update(disp=verbose)
        a_fit = optimizer(self._cost_wrapper, a0, args=(self._params,), **self._optimizer_kwargs)
        a_fit = np.array(a_fit, ndmin=1)
        self._params.ingest(a_fit)
        return self

    def predict(self, x=None):
        """
        Returns an array of predictions based on trained models.  If
        x is not supplied, than the fit over x values in training set is used
        """
        if self._raw_model is None:
            raise ValueError("No model function")
        if x is not None:
            p = self.all_params
            p.x = x  # type: ignore
        else:
            p = self._params
        return self._raw_model(p)

    def df(self, x=None):
        import pandas as pd

        p = self._params
        if x is None:
            x = p.x  # type: ignore
        y = self.predict(x)
        return pd.DataFrame({"x": x, "y": y})

    def plot(
        self,
        *,
        x=None,
        scale_factor=1,
        label=None,
        line_color=None,
        scatter_color=None,
        size=10,
        xlabel="x",
        ylabel="y",
        as_components=False,
    ):
        """
        Draw plots for model fit results.
        Params:
            x: A custom x over which to draw fits
            scale_factor: Scale all y values by this factor
            label: A string with which to label the fit
            line_color: Color for fit line
            scatter_color: Color for data points
            size: size of scatter points
            xlabel: x axis label
            ylabel: y axis label
            as_components: if True, return chart components rather than overlay
        """
        import holoviews as hv

        p = self._params
        if x is None:
            x = p.x  # type: ignore
        line_color = line_color if line_color else ezr.cc.b  # type: ignore
        scatter_color = scatter_color if scatter_color else ezr.cc.a  # type: ignore
        label_val = label if label else "Fit"
        try:
            scatter = hv.Scatter((p.x, scale_factor * p.y), xlabel, ylabel, label=label_val).opts(color=scatter_color, size=size, alpha=0.5)  # type: ignore
        except Exception:
            raise RuntimeError("You must import holoviews and set bokeh backround for plotting to work")
        line = hv.Curve((x, scale_factor * self.predict(x)), label=label_val).opts(color=line_color)
        traces = [scatter, line]
        if as_components:
            return traces
        else:
            return hv.Overlay(traces)


def classifier_evaluation_plots(
    trained_model,
    X_test,
    y_test,
    threshold=0.5,
    plots="confusion_matrix, variable_importance, roc_curve, precision_recall, lift",
):
    """
    Creates classifier evalutaion plots.
    Args:
        trained_model: A trained sklearn or xbg model
        X_test: A test dataframe of X variables
        y_test A test dataframe of y variables (0 or 1)
        threshold: The operating point of the classifier
    """
    import pandas as pd
    import numpy as np
    from sklearn.metrics import auc, average_precision_score, confusion_matrix, precision_recall_curve, roc_curve
    from IPython.display import display
    import matplotlib.pyplot as plt

    if "variable_importance" in plots:
        imp = pd.Series(trained_model.feature_importances_, index=trained_model.feature_names_in_).sort_values()
        plt.figure()
        ax = imp.plot.barh(xlabel="Variable Importance")
        ax.figure.subplots_adjust(left=0.3)
        plt.show()
    y_pred_prob = trained_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= threshold).astype(int)
    if "confusion_matrix" in plots:
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrix_df = pd.DataFrame(conf_matrix, index=["True Negative", "True Positive"], columns=["Predicted Negative", "Predicted Positive"])  # type: ignore
        conf_matrix_df["Total"] = conf_matrix_df.sum(axis=1)
        conf_matrix_df = conf_matrix_df.T
        conf_matrix_df["Total"] = conf_matrix_df.sum(axis=1)
        conf_matrix_df = conf_matrix_df.T
        conf_matrix_df.index.name = "Counts"
        display(conf_matrix_df)
        print()
        dfcr = conf_matrix_df.divide(conf_matrix_df.loc[:, "Total"], axis=0)  # type: ignore
        dfcr.index.name = "RowNorm"
        display(dfcr)
        print()
        dfcr = conf_matrix_df.divide(conf_matrix_df.loc["Total", :], axis=1)  # type: ignore
        dfcr.index.name = "ColNorm"
        display(dfcr)
    if "roc_curve" in plots:
        fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        closest_threshold_index = np.argmin(np.abs(thresholds_roc - threshold))
        operating_point_roc = (fpr[closest_threshold_index], tpr[closest_threshold_index])
        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="-")
        max_ind = min([len(fpr), len(thresholds_roc)])
        plt.plot(fpr[:max_ind], thresholds_roc[:max_ind], label="threshold (y ax)", color="grey", linestyle="--")
        plt.scatter(*operating_point_roc, color="red", label="Operating Point")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC)")
        plt.legend(loc="best")
        plt.show()
    if "precision_recall" in plots:
        precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_prob)
        average_precision = average_precision_score(y_test, y_pred_prob)
        closest_threshold_index = np.argmin(np.abs(thresholds_pr - threshold))
        operating_point_pr = (recall[closest_threshold_index], precision[closest_threshold_index])
        plt.figure()
        plt.step(recall, precision, color="b", alpha=0.2, where="post", label="precision")
        plt.fill_between(recall, precision, alpha=0.2, color="b", step="post")
        plt.scatter(*operating_point_pr, color="red", label="Operating Point")
        max_ind = min([len(recall), len(thresholds_pr)])
        plt.plot(recall[:max_ind], thresholds_pr[:max_ind], label="threshold (y ax)", color="grey", linestyle="--")
        plt.xlabel(f"Recall = {np.round(recall[closest_threshold_index], 3)}")
        plt.ylabel(f"Precision = {np.round(precision[closest_threshold_index], 3)}")
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f"Precision-Recall curve: AP={average_precision:.3f}")
        plt.legend(loc="best")
        plt.show()
    if "lift" in plots:
        lift = precision / precision[0]
        operating_point_lift = (recall[closest_threshold_index], lift[closest_threshold_index])
        plt.figure()
        plt.plot(recall, lift, color="k")
        plt.scatter(*operating_point_lift, color="red", label="Operating Point")
        plt.xlabel("Recall")
        plt.ylabel("Lift")
        plt.show()
