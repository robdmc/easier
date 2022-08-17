class Parallel:
    """
    A wrapper around native python parallel processing
    """
    def __init__(self, max_workers, kind='thread', show_progress=False):
        """
        Args:
            max_workers: The number of parallel jobs to run
            kind: "thread" or "process" determine how to parallelize
            show_progress: Display tqdm notebook-based progress bar

        Example:
            # Define the job you want to run
            def job(*args, **kwargs):
                import time
                time.sleep(args[0])
                return (args, kwargs)

            # Initialize the parallel processor
            p = ezr.Parallel(10)

            # Specify a list of args and kwargs you want to run
            call_tuples_list = [
                ([1], {'a': 1}),  # (args, kwargs)
                ([2], {'a': 2}),
                ([3], {'a': 3}),
            ]

            # Get results from a generator as jobs are completed
            for result in p.as_complete(job, call_tuples_list):
                print(result)
        """
        from concurrent import futures as futures_module
        self.futures_module = futures_module
        self.show_progress = show_progress

        if kind == 'thread':
            self.executor = self.futures_module.ThreadPoolExecutor(max_workers=max_workers)

        elif kind == 'process':

            import multiprocessing
            import warnings

            try:
                multiprocessing.set_start_method("fork")
            except RuntimeError:
                if multiprocessing.get_start_method() != 'fork':
                    warnings.warn(
                        'multiprocessing.set_start_method("fork") raised an error.  You may need to restart kernel')

            self.executor = self.futures_module.ProcessPoolExecutor(max_workers=max_workers)
        else:
            allowed_kinds = ['thread', 'process']
            raise ValueError(f'kind must be one of {allowed_kinds}')

    def as_complete(self, func, call_tuple_iterable):
        """
        Yields results as they are completed. (This is a generator).
        Each parallel call takes the args tuple and kwargs dict
        specified in an element of the call_tuple_iterable.
        Each call tuple must contain a tuple or list followed by a dict

        p = Parallel(3)
        for result in p.as_complete(my_func, [(args1, kwargs1), ((args2, kwargs2)), ...]):
            print(result)
        """
        call_tuple_list = list(call_tuple_iterable)
        futures_obj_list = []
        for (args, kwargs) in call_tuple_list:
            futures_obj = self.executor.submit(func, *args, **kwargs)
            futures_obj_list.append(futures_obj)

        if self.show_progress:
            try:
                from tqdm.notebook import tqdm
                pbar = tqdm(total=len(call_tuple_list))
            except: # noqa
                pass

        for results_obj in self.futures_module.as_completed(futures_obj_list):
            if self.show_progress:
                try:
                    pbar.update(1)
                except:  # noqa
                    pass
            yield results_obj.result()
