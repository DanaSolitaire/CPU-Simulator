#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import math

class Process:
    def __init__(self, process_id, arrival_time, burst_time, bursts, ios, cpu, lamb):
        self.process_id = process_id
        self.arrival_time = arrival_time
        self.burst_time = burst_time
        self.bursts = bursts
        self.ios = ios
        self.cpu = cpu
        self.tau = int(1/lamb)
        self.og_burst = -1
        self.blocked = 0
        self.current_burst = 0
        self.burst_start = 0
        self.arrival_queue = self.arrival_time
        self.waiting_time = 0
        self.turnaround_time = 0
    
class Rand(object):
    def __init__(self, seed):
        self.n = seed
    def srand(self, seed):
        self.n = (seed << 16) + 0x330e
    def next(self):
        self.n = (25214903917 * self.n + 11) & (2**48 - 1)
        return self.n
    def drand(self):
        return self.next() / 2**48

def next_exp(rand,lamb):
    x = -math.log(rand.drand())/lamb
    return x

def processes(n, n_cpu, seed, lamb, ceil):
    """_summary_

    Args:
        n (int): number of processes
        n_cpu (int): number of processes that are CPU-bound
        seed (int): seed for the pseudorandom number sequence
        lamb (float): rate parameter of exponential distribution
        ceil (int): upper bound for valid pseudo-random numbers
    """
    rand = Rand(0)
    rand.srand(seed)
    alph='ABCDEFGHIJKLMNOPQRSTTUVQXYZ'
    procs = []
    
    for process in range(n):
        
        num = next_exp(rand, lamb)
        
        while (num < 0) | (num > ceil):
            num = next_exp(rand, lamb)
            
        arrival_time = math.floor(num)
        bursts = math.ceil(rand.drand()*64)
        if process < (n-n_cpu):
            print(f"I/O-bound process {alph[process]}: arrival time {arrival_time}ms; {bursts} CPU bursts:")
        else:
            print(f"CPU-bound process {alph[process]}: arrival time {arrival_time}ms; {bursts} CPU bursts:")
        cpu_bursts = []
        io_bursts = []
        for burst in range(bursts):
            
            cpu_burst_time = math.ceil(next_exp(rand, lamb))
            
            while cpu_burst_time > ceil:
                cpu_burst_time = math.ceil(next_exp(rand, lamb))
                
            if burst < bursts-1:
                io_burst_time = math.ceil(next_exp(rand, lamb))
                
                while io_burst_time > ceil:
                    io_burst_time = math.ceil(next_exp(rand, lamb))
                    
                io_burst_time *= 10
                
            if process >= (n-n_cpu):
                
                cpu_burst_time *= 4
                io_burst_time = io_burst_time/8
            cpu_bursts.append(math.floor(cpu_burst_time))   
            io_bursts.append(math.floor(io_burst_time)) 
            if burst >= bursts-1:
                terminal_out(math.floor(cpu_burst_time))
            else:
                terminal_out(math.floor(cpu_burst_time),math.floor(io_burst_time))
        procs.append(Process(alph[process], arrival_time, bursts, cpu_bursts, io_bursts, process >= (n-n_cpu), lamb))
    return procs
    
def terminal_out(cpu_burst, io_burst=None):
    if not io_burst:
        print(f"--> CPU burst {cpu_burst}ms")
    else:
        print(f"--> CPU burst {cpu_burst}ms --> I/O burst {io_burst}ms")
        
def print_queue(queue):
    print("[Q",end="")
    if len(queue)==0:
        print(" <empty>]")
        return
    for p in queue:
        print(f" {p.process_id}",end="")
    print("]")
    
def calculate_tau(alpha, burst_time, prev_tau):
    tau = alpha * burst_time + (1 - alpha) * prev_tau
    return math.ceil(tau)

def cpu_utilization(processes, total_time):
    if total_time == 0:
        return 0
    return (sum(sum(p.bursts) for p in processes)/total_time)*100
    
def avg_burst(processes):
    if sum(len(p.bursts) for p in processes) == 0:
        return 0
    return math.ceil((sum(sum(p.bursts) for p in processes)/sum(len(p.bursts) for p in processes))*1000)/1000

def avg(ls):
    if len(ls)==0:
        return 0
    return math.ceil((sum(ls)/len(ls))*1000)/1000
    
def shortest_job_first(processes):
    
    processes = sorted(processes, key=lambda x: x.arrival_time)
    processes_copy = processes.copy()
    
    io_context = 0
    cpu_context = 0
    
    cpu_turnaround = []
    io_turnaround = []
    
    cpu_wait = []
    io_wait = []
    
    queue = []
    time = 0
    running = []
    
    print(f"time 0ms: Simulator started for SJF ",end="")
    print_queue(queue)
    
    while len(processes) != 0:
        
        for process in processes:
            
            if process.arrival_time <= time and process not in queue and process not in running:
                
                if process.blocked == 0:
                    
                    queue.append(process)
                    queue.sort(key = lambda x: x.tau)
                    
                    time = int(process.arrival_time)
                    
                    print(f"time {time}ms: Process {process.process_id} (tau {process.tau}ms) arrived; added to ready queue ",end="")
                    print_queue(queue)
                    process.arrival_queue = time
                    
                if process.blocked != 0 and process.blocked <= time:
                    
                    queue.append(process)
                    queue.sort(key = lambda x: x.tau)
                    
                    time = int(process.blocked)
                    
                    print(f"time {time}ms: Process {process.process_id} (tau {process.tau}ms) completed I/O; added to ready queue ",end="")
                    print_queue(queue)
                    process.arrival_queue = time
                    
        time += 1
        
        if len(running) == 0 and len(queue) != 0:
            
            time -= 1
            time += int(t_cs/2)
            
            queue[0].burst_start = time
            running.append(queue.pop(0))
            
            print(f"time {time}ms: Process {running[0].process_id} (tau {running[0].tau}ms) started using the CPU for {running[0].bursts[running[0].current_burst]}ms burst ",end="")
            print_queue(queue)
            
        if len(running)!=0:
            
            if time >= (running[0].burst_start+running[0].bursts[running[0].current_burst]):
                
                time = int(running[0].burst_start+running[0].bursts[running[0].current_burst])
                running[0].current_burst += 1
                
                if running[0].current_burst >= running[0].burst_time:
                    
                    print(f"time {time}ms: Process {running[0].process_id} terminated ",end="")
                    print_queue(queue)
                    
                    processes.remove(running[0])
                    old = running.pop(0)
                    
                    time+=int(t_cs/2)
                else:
                    
                    if (running[0].burst_time-running[0].current_burst) != 1:
                        
                        print(f"time {time}ms: Process {running[0].process_id} (tau {running[0].tau}ms) completed a CPU burst; {running[0].burst_time-running[0].current_burst} bursts to go ",end="")
                    
                    else:
                        
                        print(f"time {time}ms: Process {running[0].process_id} (tau {running[0].tau}ms) completed a CPU burst; {running[0].burst_time-running[0].current_burst} burst to go ",end="")
                    
                    print_queue(queue)
                    
                    print(f"time {time}ms: Recalculating tau for process {running[0].process_id}: old tau {running[0].tau}ms ==> new tau {calculate_tau(alpha, running[0].bursts[running[0].current_burst-1],running[0].tau)}ms ",end = "")
                    print_queue(queue)
                    
                    running[0].tau = calculate_tau(alpha, running[0].bursts[running[0].current_burst-1],running[0].tau)
                    running[0].blocked = int(running[0].ios[running[0].current_burst-1]+time+(t_cs/2))
                    
                    print(f"time {time}ms: Process {running[0].process_id} switching out of CPU; blocking on I/O until time {running[0].blocked}ms ",end="")
                    print_queue(queue)
                    
                    old = running.pop(0)
                    time += int(t_cs/2)
                    
                if old.cpu:
                    cpu_turnaround.append(time-old.arrival_queue)
                    cpu_wait.append((cpu_turnaround[-1]-t_cs)-old.bursts[old.current_burst-1])
                    cpu_context += 1
                    
                else:
                    io_turnaround.append(time-old.arrival_queue)
                    io_wait.append((io_turnaround[-1]-t_cs)-old.bursts[old.current_burst-1])
                    io_context += 1
                    
    print(f"time {time}ms: Simulator ended for SJF ", end="")
    print_queue(queue)
    
    cpu_util = cpu_utilization(processes_copy, time)
    print(f"CPU utilization: {cpu_util:.3f}")
    
    avg_burst_time_io = avg_burst(processes_copy[n_cpu:])
    avg_burst_time_cpu = avg_burst(processes_copy[:n_cpu])
    avg_burst_time = avg_burst(processes_copy)
    print(f"Average burst time: {avg_burst_time:.3f} ({avg_burst_time_cpu:.3f}/{avg_burst_time_io:.3f})")
    
    avg_cpu_turn = avg(cpu_turnaround)
    avg_io_turn = avg(io_turnaround)
    avg_turnaround = avg(cpu_turnaround+io_turnaround)
    print(f"Average turnaround time: {avg_turnaround:.3f} ({avg_cpu_turn}/{avg_io_turn})")
    
    avg_cpu_wait = avg(cpu_wait)
    avg_io_wait = avg(io_wait)
    avg_wait = avg(cpu_wait+io_wait)
    print(f"Average wait time: {math.ceil(avg_wait*1000)/1000:.3f} ({avg_cpu_wait}/{avg_io_wait})")
    
    print(f"Number of context switches: {cpu_context+io_context} ({cpu_context}/{io_context})")
    
    print("Number of preemptions: 0 (0/0)")
    
def round_robin(processes):
    
    processes = sorted(processes, key=lambda x: x.arrival_time)
    processes_copy = processes.copy()
    
    cpu_preemp = 0
    io_preemp = 0
    
    io_context = 0
    cpu_context = 0
    
    cpu_turnaround = []
    io_turnaround = []
    
    cpu_wait = []
    io_wait = []
    
    queue = []
    time = 0
    running = []
    
    print(f"time 0ms: Simulator started for RR ",end="")
    print_queue(queue)
    
    while len(processes) != 0:
        
        for process in processes:
            
            if process.arrival_time <= time and process not in queue and process not in running:
                
                if process.blocked == 0:
                    
                    queue.append(process)
                    
                    time = int(process.arrival_time)
                    
                    print(f"time {time}ms: Process {process.process_id} arrived; added to ready queue ",end="")
                    print_queue(queue)
                    
                    process.arrival_queue = time
                    
                if process.blocked != 0 and process.blocked <= time:
                    
                    queue.append(process)
                    
                    time = int(process.blocked)
                    
                    print(f"time {time}ms: Process {process.process_id} completed I/O; added to ready queue ",end="")
                    print_queue(queue)
                    
                    process.arrival_queue = time
                    
        time += 1
        
        if len(running) == 0 and len(queue) != 0:
            
            time -= 1
            time += int(t_cs/2)
            
            queue[0].burst_start = time
            running.append(queue.pop(0))
            
            if running[0].og_burst == -1:
                running[0].og_burst = running[0].bursts[running[0].current_burst]
                
                print(f"time {time}ms: Process {running[0].process_id} started using the CPU for {running[0].bursts[running[0].current_burst]}ms burst ",end="")
                print_queue(queue)
            else:
                print(f"time {time}ms: Process {running[0].process_id} started using the CPU for remaining {running[0].bursts[running[0].current_burst]}ms of {running[0].og_burst}ms burst ",end="")
                print_queue(queue)
            
        if len(running)!=0:
            
            if time >= (running[0].burst_start+t_slice) and len(queue) != 0:
                
                print(f"time {time}ms: Time slice expired; preempting process {running[0].process_id} with {running[0].bursts[running[0].current_burst]-t_slice}ms remaining ",end="")
                print_queue(queue)
                
                running[0].bursts[running[0].current_burst]-=t_slice
                
                old = running.pop(0)
                
                queue.append(old)
                
                time += int(t_cs/2)
                
                if old.cpu:
                    cpu_preemp += 1
                else:
                    io_preemp += 1
                    
                continue
            
            if time >= (running[0].burst_start+t_slice) and len(queue) == 0:
                
                print(f"time {time}ms: Time slice expired; no preemption because ready queue is empty ",end="")
                print_queue(queue)
                
                running[0].burst_start = time
                running[0].bursts[running[0].current_burst]-=t_slice
                
                time+=1
                
                continue    
            
            if time >= (running[0].burst_start+running[0].bursts[running[0].current_burst]):
                
                time = int(running[0].burst_start+running[0].bursts[running[0].current_burst])
                running[0].current_burst += 1
                
                if running[0].current_burst >= running[0].burst_time:
                    
                    print(f"time {time}ms: Process {running[0].process_id} terminated ",end="")
                    print_queue(queue)
                    
                    processes.remove(running[0])
                    old = running.pop(0)
                    
                    time+=int(t_cs/2)
                else:
                    
                    if (running[0].burst_time-running[0].current_burst) != 1:
                        
                        print(f"time {time}ms: Process {running[0].process_id} completed a CPU burst; {running[0].burst_time-running[0].current_burst} bursts to go ",end="")
                    
                    else:
                        
                        print(f"time {time}ms: Process {running[0].process_id} completed a CPU burst; {running[0].burst_time-running[0].current_burst} burst to go ",end="")
                    
                    print_queue(queue)
                    
                    running[0].tau = calculate_tau(alpha, running[0].bursts[running[0].current_burst-1],running[0].tau)
                    running[0].blocked = int(running[0].ios[running[0].current_burst-1]+time+(t_cs/2))
                    
                    print(f"time {time}ms: Process {running[0].process_id} switching out of CPU; blocking on I/O until time {running[0].blocked}ms ",end="")
                    print_queue(queue)
                    
                    running[0].og_burst = -1
                    
                    old = running.pop(0)
                    time += int(t_cs/2)
                    
                if old.cpu:
                    cpu_turnaround.append(time-old.arrival_queue)
                    cpu_wait.append((cpu_turnaround[-1]-t_cs)-old.bursts[old.current_burst-1])
                    cpu_context += 1
                    
                else:
                    io_turnaround.append(time-old.arrival_queue)
                    io_wait.append((io_turnaround[-1]-t_cs)-old.bursts[old.current_burst-1])
                    io_context += 1
        
    print(f"time {time}ms: Simulator ended for RR ", end="")
    print_queue(queue)
    
    cpu_util = cpu_utilization(processes_copy,time)
    print(f"CPU utilization: {cpu_util:.3f}")
    
    avg_burst_time_io = avg_burst(processes_copy[n_cpu:])
    avg_burst_time_cpu = avg_burst(processes_copy[:n_cpu])
    avg_burst_time = avg_burst(processes_copy)
    print(f"Average burst time: {avg_burst_time:.3f} ({avg_burst_time_cpu:.3f}/{avg_burst_time_io:.3f})")
    
    avg_cpu_turn = avg(cpu_turnaround)
    avg_io_turn = avg(io_turnaround)
    avg_turnaround = avg(cpu_turnaround+io_turnaround)
    print(f"Average turnaround time: {avg_turnaround:.3f} ({avg_cpu_turn}/{avg_io_turn})")
    
    avg_cpu_wait = avg(cpu_wait)
    avg_io_wait = avg(io_wait)
    avg_wait = avg(cpu_wait+io_wait)
    print(f"Average wait time: {math.ceil(avg_wait*1000)/1000:.3f} ({avg_cpu_wait}/{avg_io_wait})")
    
    print(f"Number of context switches: {cpu_context+io_context} ({cpu_context}/{io_context})")
    
    print(f"Number of preemptions: {cpu_preemp+io_preemp} ({cpu_preemp}/{io_preemp})")

def tau(burst, avg_burst, alpha):
    return alpha * burst + (1 - alpha) * avg_burst

if __name__ == '__main__':
    try:
        assert len(sys.argv) == 9
    except Exception as e:
        print("ERROR: Incorrect number of arguments", e, file=sys.stderr)
        sys.exit(1)
    try: 
        n = int(sys.argv[1])
        n_cpu = int(sys.argv[2])
        seed = int(sys.argv[3])
        lamb = float(sys.argv[4])
        ceil = int(sys.argv[5])
        t_cs = int(sys.argv[6])
        alpha = float(sys.argv[7])
        t_slice = int(sys.argv[8])
        
    except ValueError as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
    if n_cpu == 1:
        print(f"<<< PROJECT PART I -- process set (n={n}) with {n_cpu} CPU-bound process >>>")
    else:
        print(f"<<< PROJECT PART I -- process set (n={n}) with {n_cpu} CPU-bound processes >>>")
        
    procs = processes(n,n_cpu,seed,lamb,ceil)
    #shortest_job_first(procs)
    round_robin(procs)
    
            
    
