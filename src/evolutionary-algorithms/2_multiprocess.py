"""Multiprocessing more similar to what I need for evolution
"""
from julia.api import Julia
Julia(compiled_modules=False)
from julia import Main
import multiprocessing

if __name__=='__main__':
    Main.eval('include("EvalExample.jl")')
    different_parameters = [1,2,3,4,5]
    num_processes = 2
    pool = multiprocessing.Pool(processes=num_processes)
    jobs = pool.map_async(Main.evaluate_parameter, different_parameters)
    outs = jobs.get()
