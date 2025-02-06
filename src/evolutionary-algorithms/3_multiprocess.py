import multiprocessing
import atexit

# This is the function that will be called on each worker.
def worker_evaluate(p):
    # This will use the global variable 'evaluate_parameter'
    # which is set by the worker initializer.
    return evaluate_parameter(p)

# This initializer will run in each worker process.
def init_worker():
    from julia.api import Julia
    # Initialize Julia (make sure this happens in the worker)
    Julia(compiled_modules=False)
    from julia import Main
    # Include the Julia file which defines the function.
    Main.eval('include("EvalExample.jl")')
    # Set the global variable in the worker process.
    global evaluate_parameter
    evaluate_parameter = Main.evaluate_parameter

    # Option 1: Unregister PyJulia's shutdown hook.
    try:
        from julia import __init__ as julia_init
        atexit.unregister(julia_init._shutdown)
    except Exception:
        pass

if __name__ == '__main__':
    # Use the spawn start method so that the worker processes do not inherit
    # the already-initialized Julia runtime.
    multiprocessing.set_start_method("spawn")
    
    # Create a pool with the initializer.
    pool = multiprocessing.Pool(processes=2, initializer=init_worker)
    different_parameters = [1, 2, 3, 4, 5]
    
    # Instead of mapping evaluate_parameter (which is undefined here),
    # map the wrapper function.
    outs = pool.map(worker_evaluate, different_parameters)
    print(outs)

    # Properly close and join the pool.
    pool.close()
    pool.join()
