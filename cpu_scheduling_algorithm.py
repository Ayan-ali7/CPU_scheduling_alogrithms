import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import timeit

class Process:
    def __init__(self, pid, arrival_time, burst_time):
        """Initialize a process with the given parameters"""
        self.pid = pid
        self.arrival_time = arrival_time
        self.burst_time = burst_time
        self.remaining_time = burst_time
        self.start_time = None
        self.completion_time = None
        
    def __str__(self):
        return f"Process {self.pid}: arrival={self.arrival_time}, burst={self.burst_time}"

def generate_processes(n=10, seed=42):
    """Generate n random processes with varying arrival and burst times"""
    np.random.seed(seed)
    arrival_times = np.sort(np.random.exponential(scale=5, size=n))
    burst_times = np.random.randint(1, 20, size=n)
    
    processes = []
    for i in range(n):
        processes.append(Process(
            pid=i+1,
            arrival_time=round(arrival_times[i], 2),
            burst_time=round(burst_times[i], 2),
        ))
    
    return processes

def calculate_metrics(processes):
    """Calculate scheduling metrics for the given list of completed processes"""
    total_waiting_time = 0
    total_turnaround_time = 0
    
    for process in processes:
        waiting_time = process.start_time - process.arrival_time
        turnaround_time = process.completion_time - process.arrival_time
        total_waiting_time += waiting_time
        total_turnaround_time += turnaround_time
    
    avg_waiting_time = total_waiting_time / len(processes)
    avg_turnaround_time = total_turnaround_time / len(processes)
    
    return avg_waiting_time, avg_turnaround_time

def measure_algorithm_overhead(algorithm_func, processes, *args):
    """
    Accurately measure the overhead of a scheduling algorithm
    by running it multiple times and taking the average
    
    Parameters:
    - algorithm_func: The scheduling algorithm function to measure
    - processes: List of Process objects
    - args: Additional arguments for the algorithm function
    
    Returns:
    - algorithm_result: The result of the algorithm
    - overhead_microseconds: The overhead in microseconds
    """
    def run_algorithm():
        return algorithm_func(processes, *args)
    
    # Measure the execution time over multiple runs for more accurate measurement
    number_of_runs = 100
    total_time = timeit.timeit(run_algorithm, number=number_of_runs)
    
    # Run one more time to get the actual result
    algorithm_result = run_algorithm()
    
    # Calculate average overhead in microseconds
    overhead_microseconds = (total_time / number_of_runs) * 1_000_000
    
    return algorithm_result, overhead_microseconds

def fcfs_core(processes):
    sorted_processes = sorted(processes, key=lambda p: p.arrival_time)
    current_time = 0
    results = []
    
    for process in sorted_processes:
        p = Process(process.pid, process.arrival_time, process.burst_time)
        
        if current_time < p.arrival_time:
            current_time = p.arrival_time
        
        p.start_time = current_time
        current_time += p.burst_time
        p.completion_time = current_time
        
        results.append(p)
    
    avg_waiting_time, avg_turnaround_time = calculate_metrics(results)
    makespan = results[-1].completion_time
    
    return {
        "name": "FCFS",
        "results": results,
        "avg_waiting_time": avg_waiting_time,
        "avg_turnaround_time": avg_turnaround_time,
        "makespan": makespan,
    }

def fcfs(processes):
    result, overhead = measure_algorithm_overhead(fcfs_core, processes)
    result["overhead"] = overhead
    return result

def sjn_core(processes):
    process_copies = [Process(p.pid, p.arrival_time, p.burst_time) for p in processes]
    process_copies.sort(key=lambda p: p.arrival_time)
    
    current_time = 0
    completed = []
    ready_queue = []
    
    while process_copies or ready_queue:
        while process_copies and process_copies[0].arrival_time <= current_time:
            ready_queue.append(process_copies.pop(0))
        
        if ready_queue:
            ready_queue.sort(key=lambda p: p.burst_time)
            current_process = ready_queue.pop(0)
            
            current_process.start_time = current_time
            current_time += current_process.burst_time
            current_process.completion_time = current_time
            
            completed.append(current_process)
        else:
            if process_copies:
                current_time = process_copies[0].arrival_time
    
    avg_waiting_time, avg_turnaround_time = calculate_metrics(completed)
    makespan = completed[-1].completion_time
    
    return {
        "name": "SJN",
        "results": completed,
        "avg_waiting_time": avg_waiting_time,
        "avg_turnaround_time": avg_turnaround_time,
        "makespan": makespan,
    }

def sjn(processes):
    result, overhead = measure_algorithm_overhead(sjn_core, processes)
    result["overhead"] = overhead
    return result

def round_robin_core(processes, time_quantum=4):
    process_copies = [Process(p.pid, p.arrival_time, p.burst_time) for p in processes]
    process_copies.sort(key=lambda p: p.arrival_time)
    
    current_time = 0
    completed = []
    ready_queue = deque()
    execution_log = []
    
    while process_copies or ready_queue:
        while process_copies and process_copies[0].arrival_time <= current_time:
            p = process_copies.pop(0)
            ready_queue.append(p)
            if p.start_time is None:
                p.start_time = current_time
        
        if ready_queue:
            current_process = ready_queue.popleft()
            
            if current_process.remaining_time <= time_quantum:
                execution_time = current_process.remaining_time
                current_time += execution_time
                current_process.remaining_time = 0
                current_process.completion_time = current_time
                completed.append(current_process)
            else:
                execution_time = time_quantum
                current_time += execution_time
                current_process.remaining_time -= time_quantum
                
                while process_copies and process_copies[0].arrival_time <= current_time:
                    p = process_copies.pop(0)
                    ready_queue.append(p)
                    if p.start_time is None:
                        p.start_time = current_time
                
                ready_queue.append(current_process)
            
            execution_log.append((current_process.pid, current_time - execution_time, current_time))
        else:
            if process_copies:
                current_time = process_copies[0].arrival_time
    
    avg_waiting_time, avg_turnaround_time = calculate_metrics(completed)
    makespan = completed[-1].completion_time
    
    return {
        "name": "Round Robin",
        "results": completed,
        "avg_waiting_time": avg_waiting_time,
        "avg_turnaround_time": avg_turnaround_time,
        "makespan": makespan,
        "execution_log": execution_log
    }

def round_robin(processes, time_quantum=4):
    result, overhead = measure_algorithm_overhead(round_robin_core, processes, time_quantum)
    result["overhead"] = overhead
    return result

def visualize_results(results):

    names = [result["name"] for result in results]
    waiting_times = [result["avg_waiting_time"] for result in results]
    turnaround_times = [result["avg_turnaround_time"] for result in results]
    makespans = [result["makespan"] for result in results]
    overheads = [result["overhead"] for result in results] 
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    axs[0, 0].bar(names, waiting_times, color='skyblue')
    axs[0, 0].set_title('Average Waiting Time')
    axs[0, 0].set_ylabel('Time')
    
    axs[0, 1].bar(names, turnaround_times, color='lightgreen')
    axs[0, 1].set_title('Average Turnaround Time')
    axs[0, 1].set_ylabel('Time')

    axs[1, 0].bar(names, makespans, color='salmon')
    axs[1, 0].set_title('Total Execution Time (Makespan)')
    axs[1, 0].set_ylabel('Time')
    
    axs[1, 1].bar(names, overheads, color='purple')
    axs[1, 1].set_title('Algorithm Overhead')
    axs[1, 1].set_ylabel('Time (microseconds)')
    

    for i, v in enumerate(overheads):
        axs[1, 1].text(i, v + 0.5, f"{v:.2f} μs", ha='center')
    
    plt.tight_layout()
    plt.show()

def print_results(results):
    for result in results:
        print(f"\n{result['name']} Results:")
        print(f"Average Waiting Time: {result['avg_waiting_time']:.2f}")
        print(f"Average Turnaround Time: {result['avg_turnaround_time']:.2f}")
        print(f"Total Execution Time (Makespan): {result['makespan']:.2f}")
        print(f"Algorithm Overhead: {result['overhead']:.4f} microseconds")
        
        print("\nProcess Execution Details:")
        for process in result['results']:
            print(f"Process {process.pid}: Start={process.start_time:.2f}, Complete={process.completion_time:.2f}, "
                  f"Wait={(process.start_time - process.arrival_time):.2f}, "
                  f"Turnaround={(process.completion_time - process.arrival_time):.2f}")

def run_simulation(num_processes=10, time_quantum=4):

    processes = generate_processes(num_processes)
    
    print(f"Generated {num_processes} processes:")
    for p in processes:
        print(f"Process {p.pid}: arrival={p.arrival_time:.2f}, burst={p.burst_time:.2f}")
    

    fcfs_result = fcfs(processes)
    sjn_result = sjn(processes)
    rr_result = round_robin(processes, time_quantum)
    
    results = [fcfs_result, sjn_result, rr_result]
    
    print_results(results)
    visualize_results(results)
    
    return results

if __name__ == "__main__":
    results = run_simulation(num_processes=20, time_quantum=4)