
import multiprocessing
import time

def worker(worker_id):
    """Function that each worker process runs."""
    print(f"Worker {worker_id} started")
    time.sleep(2)  # Simulate some work
    print(f"Worker {worker_id} finished")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # Ensure safe multiprocessing

    num_workers = 10
    processes = []

    # Create and start 10 worker processes, each with a unique ID (1-10)
    for i in range(1, num_workers + 1):
        p = multiprocessing.Process(target=worker, args=(i,))
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print("All workers have finished.")

