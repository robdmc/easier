def tvd(y, lamda, num_iter=20, return_cost=False):
    """
    Performs total variation denoising on a supplied numpy array
    Args:
        lamda:
            The L1 regularization parameter.  A reaonable starting
            value for this is something like the standard devation of
            your noise.

        num_iter:
            This is an iterative algorithm.
            You may need to alter this number for good results

        return_cost:
            Set this to true if you also want to return a cost for each iteration
    Reurns:
        If return_cost is False
           - a numpy array of filtered results
        if return_ost is True
           (filtered_results, cost_array)
           - a numpy array of filtered results
           - a numpy array of lengh(num_iter) holding the cost at each iteration


    Implements algorithm written up by Ivan Selesnick.
    As of Sept. 2021 the paper was available from this url

    https://eeweb.engineering.nyu.edu/iselesni/lecture_notes/TVDmm/TVDmm.pdf

    It is also included in the references directory of this repository.
    """
    import numpy as np
    from scipy import sparse
    from scipy.sparse import linalg  # noqa weird scipy import thing

    if not isinstance(y, np.ndarray):
        raise ValueError('tvd can only be used on numpy arrays')

    if not len(y.shape) == 1:
        raise ValueError('tvd can only be used on flat arrays. Try y.flatten()')

    if np.any(np.isnan(y)):
        raise ValueError('tvd must not contain any nans')

    N = len(y)
    y = np.expand_dims(y, -1)
    cost = np.zeros(num_iter)
    I = sparse.identity(N, format='csr')  # noqa
    D = I[1:N, :] - I[:N - 1, :]
    DDT = D @ D.T

    x = y
    Dx = D @ x
    Dy = D @ y
    for k in range(num_iter):
        F = sparse.diags(np.abs(Dx.flatten()) / lamda, format='csr') + DDT
        x = y - np.expand_dims(D.T @ sparse.linalg.spsolve(F, Dy), -1)
        Dx = D @ x
        cost[k] = .5 * np.sum((x - y) ** 2) + lamda * np.sum(np.abs(Dx))

    if return_cost:
        return x.flatten(), cost
    else:
        return x.flatten()
