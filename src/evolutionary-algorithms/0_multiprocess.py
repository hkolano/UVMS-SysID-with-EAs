"""This is a multithreading test. If I have two functions run in seperate threads, but both functions modify a global variable in Julia,
then does that global variable get modified in both threads in a way that we see reflected in the main thread?
OR
Does each thread make a copy of that global variable and modify that copy?
"""
from pathlib import Path
import os
from julia.api import Julia
Julia(compiled_modules=False)
from julia import Main
import numpy as np
import os
import multiprocessing

def add_number(num):
    # Main.num = num
    y = Main.x+num
    return y
    
if __name__=='__main__':
    Main.eval("x=0")

    num_threads = 2
    pool = multiprocessing.Pool(processes=num_threads)

    numbers = [1,1]

    jobs = pool.map_async(add_number, numbers)
    outs = jobs.get()

    print(Main.x)
