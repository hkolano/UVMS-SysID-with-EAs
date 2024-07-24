from guppy import hpy

def get_memory_usage(h):
    memory_bytes = int(str(h.heap()).split('\n')[0].split(' ')[-2])
    return memory_bytes

