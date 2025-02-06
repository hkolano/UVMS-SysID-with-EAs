from pathlib import Path
import os
from julia.api import Julia
Julia(compiled_modules=False)
from julia import Main
import numpy as np
import os
import multiprocessing

def add_number(num):
    Main.num = num
    Main.eval('x = x+num')
    return Main.x
    
if __name__=='__main__':
    Main.eval('x=0')

    for num in [1,1]:
        x = add_number(1)
    
    print(x)