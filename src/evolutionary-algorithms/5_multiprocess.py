import multiprocessing
import time
import random

# Global variable to store the worker ID
WORKER_ID = None

def init_worker():
    """Assign a unique worker ID based on process identity."""
    global WORKER_ID
    WORKER_ID = multiprocessing.current_process()._identity[0]  # Unique ID (1 to num_workers)
    print(f"Worker {WORKER_ID} initialized.")

def evaluate(individual):
    """Function to evaluate an individual using the assigned worker ID."""
    global WORKER_ID
    time.sleep(random.uniform(0.5, 1.5))  # Simulate computation time
    return f"Worker {WORKER_ID} evaluated individual {individual}"

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # Ensure safe multiprocessing

    num_workers = 10
    with multiprocessing.Pool(processes=num_workers, initializer=init_worker) as pool:
        
        num_generations = 5
        population = list(range(20))  # Example population of 20 individuals

        for gen in range(num_generations):
            print(f"\n--- Generation {gen + 1} ---")

            # Distribute evaluation using map_async
            result = pool.map_async(evaluate, population)
            evaluated_population = result.get()  # Wait for results
            
            # Print evaluation results
            for res in evaluated_population:
                print(res)

    print("All generations completed.")
