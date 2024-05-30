"""
2023.08.30
author: femke.hurtak@epfl.ch
Helping script for working with multiprocessing
"""

import time
import multiprocessing as mp


def queue_wrapping(function, result_queue, **kwargs):
    """
    Wrapper function to allow a function to be called and add the result to a queue.
    This allows to calla function compatible with multiprocessing.

    Usage:
        > result_queue = mp.Queue()
        > args = kwargs
        > args["result_queue"] = result_queue
        > args["function"] = your_stochastic_function
        > process = mp.Process(target=queue_wrapping, kwargs=args)
        > process.start()
        > process.join()
        > result = result_queue.get()
    """
    result = function(**kwargs)
    result_queue.put(result)


def run_function_with_timeout(target, args, time_limit):
    """
    Run a function with a time limit. If the function takes longer than the time limit,
    the function is terminated and restarted, until it runs within the time limit.
    """
    success = False
    while not success:
        start_time = time.time()

        # Create a Queue to get the return value from the function
        result_queue = mp.Queue()
        # reshape the function call to comply with Queue
        args["result_queue"] = result_queue
        args["function"] = target

        # Create a process for running your_stochastic_function
        process = mp.Process(
            target=queue_wrapping,
            kwargs=args,
        )

        # Start the process
        process.start()

        # Wait for the process to finish or time out
        process.join(timeout=time_limit)

        # Check if the process is still alive (i.e., function didn't terminate)
        if process.is_alive():
            process.terminate()
            process.join()  # Wait for the process to be properly terminated before restarting
        else:
            # elapsed_time = time.time() - start_time
            result = result_queue.get()
            success = True  # Set to True to break out of the while loop
            break
    return result
