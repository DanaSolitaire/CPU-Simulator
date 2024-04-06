#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import math
import copy

class Process:
    def __init__(self, process_id, arrival_time, burst_time, cpu_bursts, io_bursts, cpu, lamb):
        self.process_id = process_id
        self.arrival_time = arrival_time
        self.burst_time = burst_time
        self.cpu_bursts = cpu_bursts
        self.io_bursts = io_bursts
        self.cpu = cpu
        self.tau = int(1/lamb)
        self.og_burst = -1
        self.blocked = -1
        self.current_burst = 0
        self.burst_start = 0
        self.preemps = 1
        self.arrival_queue = self.arrival_time
    
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
        #prints output for project 1
        
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
        ...
        #print(f"--> CPU burst {cpu_burst}ms")
    else:
        ...
        #print(f"--> CPU burst {cpu_burst}ms --> I/O burst {io_burst}ms")
        
def print_queue(queue):
    """prints queue\n
    precede with print(f"...", end="")

    Args:
        queue (list<Process>): queue for scheduling algorithm
    """
    print("[Q",end="")
    if len(queue)==0:
        print(" <empty>]")
        return
    for p in queue:
        print(f" {p.process_id}",end="")
    print("]")
    
def calculate_tau(alpha, burst_time, prev_tau):
    """updates tau based on actual burst time

    Args:
        alpha (float): smoothing factor
        burst_time (int): actual burst time
        prev_tau (int): previous tau estimation

    Returns:
        int: updated tau
    """
    tau = alpha * burst_time + (1 - alpha) * prev_tau
    return math.ceil(tau)

def simout_write(alg,processes,time,cpu_turnaround,io_turnaround,cpu_wait,io_wait,cpu_context,io_context,cpu_preemp,io_preemp):
    """function to calculate measurements and write to simout...call at the end of each scheduling algorithm

    Args:
        alg (string): 'FCFS', 'SJF', 'SRT', or 'RR'
        processes (list<Process>): list of processes
        time (int): final time
        cpu_turnaround (list<int>): list of turnaround times for each burst in the CPU-bound processes
        io_turnaround (list<int>): list of turnaround times for each burst in the IO-bound processes
        cpu_wait (list<int>): list of wait times for each burst in the CPU-bound processes
        io_wait (list<int>): list of wait times for each burst in the IO-bound processes
        cpu_context (int): number of context switches for CPU-bound processes
        io_context (int): number of context switches for IO-bound processes
        cpu_preemp (int): number of preemptions for CPU-bound processes
        io_preemp (int): number of preemptions for IO-bound processes
    """
    
    def cpu_utilization(processes, total_time):
        if total_time == 0:
            return 0
        return math.ceil((sum(sum(p.cpu_bursts) for p in processes)/total_time)*100000)/1000
        
    def avg_burst(processes):
        if sum(len(p.cpu_bursts) for p in processes) == 0:
            return 0
        return math.ceil((sum(sum(p.cpu_bursts) for p in processes)/sum(len(p.cpu_bursts) for p in processes))*1000)/1000

    def avg(ls):
        if len(ls)==0:
            return 0
        return math.ceil((sum(ls)/len(ls))*1000)/1000
    
    cpu_util = cpu_utilization(processes, time)
    avg_burst_time_io = avg_burst(processes[:n_cpu+1])
    avg_burst_time_cpu = avg_burst(processes[n_cpu+1:])
    avg_burst_time = avg_burst(processes)
    avg_cpu_turn = avg(cpu_turnaround)
    avg_io_turn = avg(io_turnaround)
    avg_turnaround = avg(cpu_turnaround+io_turnaround)
    avg_cpu_wait = avg(cpu_wait)
    avg_io_wait = avg(io_wait)
    avg_wait = avg(cpu_wait+io_wait)
    
    simout.write(f"\nAlgorithm {alg}\n")
    simout.write(f"-- CPU utilization: {cpu_util:.3f}%\n")
    simout.write(f"-- average CPU burst time: {avg_burst_time:.3f} ms ({avg_burst_time_cpu:.3f} ms/{avg_burst_time_io:.3f} ms)\n")
    simout.write(f"-- average wait time: {math.ceil(avg_wait*1000)/1000:.3f} ms ({avg_cpu_wait} ms/{avg_io_wait} ms)\n")
    simout.write(f"-- average turnaround time: {avg_turnaround:.3f} ms ({avg_cpu_turn} ms/{avg_io_turn} ms)\n")
    simout.write(f"-- number of context switches: {cpu_context+io_context} ({cpu_context}/{io_context})\n")
    simout.write(f"-- number of preemptions: {cpu_preemp+io_preemp} ({cpu_preemp}/{io_preemp})\n")

def first_come_first_serve(processes):
    processes_copy = copy.deepcopy(processes)
    processes_copy = sorted(processes_copy, key=lambda x: x.arrival_time)
    io_blocked = []
    start_times = []
    queue = []
    time = 0
    running = []

    old = Process(".",1,1,[0],[0],False,1)
    
    print(f"time 0ms: Simulator started for FCFS ",end="")
    print_queue(queue)
    
    #While there are still processes to run
    while len(processes_copy) != 0:
        
        for process in processes_copy:
            if process.arrival_time == time and process not in queue and process not in running:
                if process.blocked > time:
                    queue.append(process)
                    if time < 10000:
                        print(f"time {time}ms: Process {process.process_id} arrived; added to ready queue ",end="")
                        print_queue(queue)
                    process.arrival_queue = time
                    
                if process.blocked != 0 and process.blocked <= time:
                    queue.append(process)
                    if time < 10000:
                        print(f"time {time}ms: Process {process.process_id} completed I/O; added to ready queue ",end="")
                        print_queue(queue)
                    process.arrival_queue = time
            
        time += 1
        #If no process is running and the queue is not empty
        if len(running) == 0 and len(queue) != 0:
            time -= 1
            time += int(t_cs/2)
            queue[0].burst_start = time
            running.append(queue.pop(0))
            running[0].cpu = True
            running[0].blocked = int(running[0].io_bursts[running[0].current_burst]+time+(t_cs/2))
            start_times.append([time,running[0]])
            if time < 10000:
                print(f"time {time}ms: Process {running[0].process_id} started using the CPU for {running[0].cpu_bursts[running[0].current_burst]}ms burst ",end="")
                print_queue(queue)
        
        #If a process is running
        if len(running)!=0:
            if running[0].current_burst+1 >= running[0].burst_time and time == (running[0].burst_start+running[0].cpu_bursts[running[0].current_burst]):
                    print(f"time {time}ms: Process {running[0].process_id} terminated ",end="")
                    print_queue(queue)
                    processes_copy.remove(running[0])
                    running.pop(0)

            elif time == (running[0].burst_start+running[0].cpu_bursts[running[0].current_burst]):
                running[0].current_burst += 1
                if time < 10000:
                    print(f"time {time}ms: Process {running[0].process_id} completed a CPU burst; {running[0].burst_time-running[0].current_burst} burst to go ",end="")
                    print_queue(queue)
                    print(f"time {time}ms: Process {running[0].process_id} switching out of CPU; blocking on I/O until time {int(running[0].io_bursts[running[0].current_burst-1]+time+(t_cs/2))}ms ",end="")
                    print_queue(queue)
                io_blocked.append([running[0],int(running[0].io_bursts[running[0].current_burst-1]+time+(t_cs/2))])

                old = running.pop(0)
                time += int(t_cs/2)

            for process in io_blocked:
                if process[1] == time:
                    queue.append(process[0])
                    if time < 10000:
                        print(f"time {time}ms: Process {process[0].process_id} completed I/O; added to ready queue ",end="")
                        print_queue(queue)
                    io_blocked.remove(process)

        elif len(running)==0 and len(queue) == 0:
            for process in io_blocked:
                if process[1] == time:
                    queue.append(process[0])
                    if time < 10000:
                        print(f"time {time}ms: Process {process[0].process_id} completed I/O; added to ready queue ",end="")
                        print_queue(queue)
                    io_blocked.remove(process)

    time += int(t_cs/2)
    print(f"time {time}ms: Simulator ended for FCFS ", end="")
    print_queue(queue)

def shortest_remaining_time(processes):
    '''
    for process in processes_copy:
        print(process.process_id)
        print(f"ARRIVAL {process.arrival_time}\n")
        print(f"Burst Time {process.burst_time}\n")
        print(f"CPU Bursts {process.cpu_bursts}\n")
        print(f"IO Bursts {process.io_bursts}\n")
        print(process.cpu)
        print(process.tau)
        print(f"OG Burst {process.og_burst}\n")
        print(process.og_burst)
        print(process.blocked)
        print(process.current_burst)
        print(process.burst_start)
        print(process.preemps)
        print(process.arrival_queue)
        print("\n")
   '''

    processes_copy = copy.deepcopy(processes)
    processes_copy = sorted(processes_copy, key=lambda x: x.arrival_time)
    
    io_context = 0
    cpu_context = 0
    premptions = 0
    io_blocked = []
    
    cpu_turnaround = []
    io_turnaround = []
    start_times = []
    
    cpu_wait = []
    io_wait = []
    
    queue = []
    time = 0
    running = []

    #sort processes by arrival time
    def srt_sort(item):
        return (item.tau, item.process_id)
    '''
    #Premption
    remaining_time = time-running[0].cpu_bursts[running[0].current_burst]
    process.tau = calculate_tau(alpha, process[0].cpu_bursts[process[0].current_burst-1],process[0].tau)
    if remaining_time < process[0].tau:
        premptions += 1
        queue.append(running[0])
        queue.sort(key = srt_sort)
        old = running.pop(0)
        old.cpu = False
        running.pop(0)
        running.append(process)
        running[0].cpu = True
    '''

    
    print(f"time 0ms: Simulator started for SRT ",end="")
    print_queue(queue)
#running[0].blocked = int(running[0].io_bursts[running[0].current_burst-1]+time+(t_cs/2))
                    
    while len(processes_copy) != 0:
        for process in processes_copy:
            if process.arrival_time == time and process not in queue and process not in running:
                if process.blocked > time:
                    queue.append(process)
                    if time < 10000:
                        print(f"time {time}ms: Process {process.process_id} arrived; added to ready queue ",end="")
                        print_queue(queue)
                    process.arrival_queue = time
                    
                if process.blocked != 0 and process.blocked <= time:
                    queue.append(process)
                    if time < 10000:
                        print(f"time {time}ms: Process {process.process_id} completed I/O; added to ready queue ",end="")
                        print_queue(queue)
                    process.arrival_queue = time
        '''
        for process in processes_copy:
            if process.arrival_time == time and process not in queue and process not in running:
                if process.blocked == 0:
                    queue.append(process)
                    queue.sort(key = srt_sort)
                    if time < 10000:
                        print(f"time {time}ms: Process {process.process_id} (tau {process.tau}ms) arrived; added to ready queue ",end="")
                        print_queue(queue)
                    process.arrival_queue = time








                    
                if process.blocked != 0 and process.blocked <= time:
                    queue.append(process)
                    queue.sort(key = srt_sort)
                    if time < 10000:
                        print(f"time {time}ms: Process {process.process_id} (tau {process.tau}ms) completed I/O; added to ready queue ",end="")
                        print_queue(queue)
                    process.arrival_queue = time
        '''
            
        time += 1
        #If no process is running and the queue is not empty
        if len(running) == 0 and len(queue) != 0:
            time -= 1
            time += int(t_cs/2)
            queue[0].burst_start = time
            running.append(queue.pop(0))
            running[0].cpu = True
            running[0].blocked = int(running[0].io_bursts[running[0].current_burst]+time+(t_cs/2))
            start_times.append([time,running[0]])
            if time < 10000:
                print(f"time {time}ms: Process {running[0].process_id} started using the CPU for {running[0].cpu_bursts[running[0].current_burst]}ms burst ",end="")
                print_queue(queue)
        
        #If a process is running
        if len(running)!=0:
            if running[0].current_burst+1 >= running[0].burst_time and time == (running[0].burst_start+running[0].cpu_bursts[running[0].current_burst]):
                    print(f"time {time}ms: Process {running[0].process_id} terminated ",end="")
                    print_queue(queue)
                    processes_copy.remove(running[0])
                    running.pop(0)

            elif time == (running[0].burst_start+running[0].cpu_bursts[running[0].current_burst]):
                running[0].current_burst += 1
                if time < 10000:
                    print(f"time {time}ms: Process {running[0].process_id} completed a CPU burst; {running[0].burst_time-running[0].current_burst} burst to go ",end="")
                    print_queue(queue)
                    print(f"time {time}ms: Process {running[0].process_id} switching out of CPU; blocking on I/O until time {int(running[0].io_bursts[running[0].current_burst-1]+time+(t_cs/2))}ms ",end="")
                    print_queue(queue)

                running[0].blocked = int(running[0].io_bursts[running[0].current_burst]+time+(t_cs/2))
                io_blocked.append([running[0],int(running[0].io_bursts[running[0].current_burst-1]+time+(t_cs/2))])

                old = running.pop(0)
                time += int(t_cs/2)

            for process in io_blocked:
                if process[1] == time:
                    queue.append(process[0])
                    queue.sort(key = srt_sort)
                    if time < 10000:
                        print(f"time {time}ms: Process {process[0].process_id} completed I/O; added to ready queue ",end="")
                        print_queue(queue)
                    running[0].blocked = 0
                    io_blocked.remove(process)

        elif len(running)==0 and len(queue) == 0:
            for process in io_blocked:
                if process[1] == time:
                    queue.append(process[0])
                    queue.sort(key = srt_sort)
                    if time < 10000:
                        print(f"time {time}ms: Process {process[0].process_id} completed I/O; added to ready queue ",end="")
                        print_queue(queue)
                    io_blocked.remove(process)

    time += int(t_cs/2)
    

    
    
    


    
    
    print(f"time {time}ms: Simulator ended for SRT ", end="")
    print_queue(queue)

    simout_write("SRT",processes,time,cpu_turnaround,io_turnaround,cpu_wait,io_wait,cpu_context,io_context,0,0)
    
    

def shortest_job_first(processes):
    def sjf_sort(item):
        return (item.tau, item.process_id)
    processes_copy = copy.deepcopy(processes)
    processes_copy = sorted(processes_copy, key=lambda x: x.arrival_time)
    
    io_queue = []
    
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
    
    while len(processes_copy) != 0:
            
        if len(running)!=0:
            
            if time == (running[0].burst_start+running[0].cpu_bursts[running[0].current_burst]):
                
                running[0].current_burst += 1
                
                if running[0].current_burst >= running[0].burst_time:
                    
                    print(f"time {time}ms: Process {running[0].process_id} terminated ",end="")
                    print_queue(queue)
                    
                    processes_copy.remove(running[0])
                    old = running.pop(0)
                    
                    if len(queue) > 0:
                        queue[0].blocked=time+t_cs
                else:
                    if time <= 9999:
                        if (running[0].burst_time-running[0].current_burst) != 1:
                            
                            print(f"time {time}ms: Process {running[0].process_id} (tau {running[0].tau}ms) completed a CPU burst; {running[0].burst_time-running[0].current_burst} bursts to go ",end="")
                        
                        else:
                            
                            print(f"time {time}ms: Process {running[0].process_id} (tau {running[0].tau}ms) completed a CPU burst; {running[0].burst_time-running[0].current_burst} burst to go ",end="")
                        
                        print_queue(queue)
                        
                        print(f"time {time}ms: Recalculating tau for process {running[0].process_id}: old tau {running[0].tau}ms ==> new tau {calculate_tau(alpha, running[0].cpu_bursts[running[0].current_burst-1],running[0].tau)}ms ",end = "")
                        print_queue(queue)
                    
                    running[0].tau = calculate_tau(alpha, running[0].cpu_bursts[running[0].current_burst-1],running[0].tau)
                    running[0].blocked = int(running[0].io_bursts[running[0].current_burst-1]+time+(t_cs/2))
                    
                    if time <= 9999:
                        print(f"time {time}ms: Process {running[0].process_id} switching out of CPU; blocking on I/O until time {running[0].blocked}ms ",end="")
                        print_queue(queue)
                    
                    
                    old = running.pop(0)
                    io_queue.append(old)
                    
                    if len(queue)>0:
                        queue[0].blocked=time+t_cs
                    
                if old.cpu:
                    cpu_turnaround.append((time+(t_cs))-old.arrival_queue)
                    cpu_wait.append((cpu_turnaround[-1]-t_cs)-old.cpu_bursts[old.current_burst-1])
                    cpu_context += 1
                    
                else:
                    io_turnaround.append((time+(t_cs))-old.arrival_queue)
                    io_wait.append((io_turnaround[-1]-t_cs)-old.cpu_bursts[old.current_burst-1])
                    io_context += 1
                continue    
        if len(running) == 0 and len(queue) != 0 and time == queue[0].blocked:
            
            queue[0].burst_start = time
            running.append(queue.pop(0))
            if time <= 9999:
                print(f"time {time}ms: Process {running[0].process_id} (tau {running[0].tau}ms) started using the CPU for {running[0].cpu_bursts[running[0].current_burst]}ms burst ",end="")
                print_queue(queue)
                            
        for process in io_queue:
            if process not in queue and process not in running:
                if process.blocked == time:
                    process.arrival_queue = time+(t_cs)/2
                    process.og_burst = -1
                    
                    queue.append(process)
                    
                    queue[-1].blocked = time + (t_cs/2)
                    
                    queue.sort(key = sjf_sort)
                    if time <= 9999:
                        print(f"time {time}ms: Process {process.process_id} (tau {process.tau}ms) completed I/O; added to ready queue ",end="")
                    if len(queue) > 1 and len(running) == 0 and queue[0].process_id != process.process_id:
                        if time <= 9999:
                            print_queue(queue[1:])
                    elif len(queue) > 1 and len(running) == 0 and queue[0].process_id == process.process_id:
                        if time <= 9999:
                            print_queue([queue[0]]+queue[2:])
                        queue[0],queue[1] = queue[1],queue[0]
                    else:
                        if time <= 9999:
                            print_queue(queue)
                    
                    io_queue.remove(process)
        for process in processes_copy: #check if any process needs to be added to ready queue
            if process.blocked == time and process not in io_queue and process not in queue and process not in running:
                    queue.append(process)
                    queue.sort(key = sjf_sort)
            if process.arrival_time == time and process not in queue and process not in running:
                    
                process.arrival_queue = time+(t_cs)/2
                process.blocked = time+(t_cs)/2
                queue.append(process)
                queue.sort(key = sjf_sort)
                print(f"time {time}ms: Process {process.process_id} (tau {queue[-1].tau}ms) arrived; added to ready queue ",end="")
                print_queue(queue)
        time += 1
        
    time += int(t_cs/2)                 
    print(f"time {time}ms: Simulator ended for SJF ", end="")
    print_queue(queue)

    simout_write("SJF",processes,time,cpu_turnaround,io_turnaround,cpu_wait,io_wait,cpu_context,io_context,0,0)    
    
def round_robin(processes):
    
    processes_copy = copy.deepcopy(processes)
    
    processes_copy = sorted(processes_copy, key=lambda x: x.arrival_time)
    
    io_queue = []
    
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
    
    while len(processes_copy) != 0:
        
        if len(running)!=0:
            if time == (running[0].burst_start+t_slice) and len(queue) == 0: #if quantum expires but no other processes are available, continue with current process
                
                if time <= 9999:
                    
                    print(f"time {time}ms: Time slice expired; no preemption because ready queue is empty ",end="")
                    print_queue(queue)
                    
                running[0].burst_start = time
                running[0].cpu_bursts[running[0].current_burst]-=t_slice
            
            if time == (running[0].burst_start+running[0].cpu_bursts[running[0].current_burst]):
                running[0].current_burst += 1
                
                if running[0].current_burst >= running[0].burst_time:
                    
                    print(f"time {time}ms: Process {running[0].process_id} terminated ",end="")
                    print_queue(queue)
                    
                    processes_copy.remove(running[0])
                    old = running.pop(0)
                    if len(queue) > 0:
                        queue[0].blocked=time+t_cs

                else:
                    if time <= 9999:
                    
                        if (running[0].burst_time-running[0].current_burst) != 1:
                            
                            print(f"time {time}ms: Process {running[0].process_id} completed a CPU burst; {running[0].burst_time-running[0].current_burst} bursts to go ",end="")
                        
                        else:
                            
                            print(f"time {time}ms: Process {running[0].process_id} completed a CPU burst; {running[0].burst_time-running[0].current_burst} burst to go ",end="")
                        
                        print_queue(queue)
                    
                    running[0].tau = calculate_tau(alpha, running[0].cpu_bursts[running[0].current_burst-1],running[0].tau)
                    running[0].blocked = int(running[0].io_bursts[running[0].current_burst-1]+time+(t_cs/2))
                    
                    if time <= 9999:
                        print(f"time {time}ms: Process {running[0].process_id} switching out of CPU; blocking on I/O until time {running[0].blocked}ms ",end="")
                        print_queue(queue)
                    
                    old = running.pop(0)
                    io_queue.append(old)
                    
                    if len(queue)>0:
                        queue[0].blocked=time+t_cs
                    
                if old.cpu:
                    cpu_turnaround.append((time+(t_cs))-old.arrival_queue)
                    cpu_wait.append((cpu_turnaround[-1]-(t_cs*old.preemps))-old.og_burst)
                    cpu_context += 1
                    
                else:
                    io_turnaround.append((time+(t_cs))-old.arrival_queue)
                    io_wait.append((io_turnaround[-1]-(t_cs*old.preemps))-old.og_burst)
                    io_context += 1
                continue
            if time == (running[0].burst_start+t_slice) and len(queue) != 0: #if a process has been running for the designated quantum, remove from running and add back to ready queue
                if time <= 9999:
                    print(f"time {time}ms: Time slice expired; preempting process {running[0].process_id} with {running[0].cpu_bursts[running[0].current_burst]-t_slice}ms remaining ",end="")
                    print_queue(queue)
                
                running[0].cpu_bursts[running[0].current_burst]-=t_slice
                running[0].blocked = time+(t_cs/2)
                old = running.pop(0)
                queue[0].blocked=time+t_cs
                
                if old.cpu:
                    cpu_preemp += 1
                    cpu_context += 1
                    
                else:
                    io_preemp += 1
                    io_context += 1
        
        if len(running) == 0 and len(queue) != 0 and time == queue[0].blocked: #if processes in ready queue and nothing is running, start running first process in queue
            queue[0].burst_start = time
            running.append(queue.pop(0))
            
            if running[0].og_burst == -1:
                
                running[0].preemps = 1
                running[0].og_burst = running[0].cpu_bursts[running[0].current_burst]
                
                if time <= 9999:
                    print(f"time {time}ms: Process {running[0].process_id} started using the CPU for {running[0].cpu_bursts[running[0].current_burst]}ms burst ",end="")
                    print_queue(queue)
            else:
                
                running[0].preemps += 1
                
                if time <= 9999:
                    print(f"time {time}ms: Process {running[0].process_id} started using the CPU for remaining {running[0].cpu_bursts[running[0].current_burst]}ms of {running[0].og_burst}ms burst ",end="")
                    print_queue(queue)
        
        for process in io_queue:
            if process not in queue and process not in running:
                if process.blocked == time:
                    process.arrival_queue = time+(t_cs)/2
                    process.og_burst = -1
                    
                    queue.append(process)
                    
                    if time <= 9999:
                        print(f"time {time}ms: Process {process.process_id} completed I/O; added to ready queue ",end="")
                        if len(queue) != 0 and len(running) == 0:
                            print_queue(queue[1:])
                        else:
                            print_queue(queue)

                    queue[-1].blocked = time + (t_cs/2)
                    
                    io_queue.remove(process)
        
        for process in processes_copy: #check if any process needs to be added to ready queue
            if process.blocked == time and process not in io_queue and process not in queue and process not in running:
                    queue.append(process)
            if process.arrival_time == time and process not in queue and process not in running:
                    
                process.arrival_queue = time+(t_cs)/2
                process.blocked = time+(t_cs)/2
                queue.append(process)
                print(f"time {time}ms: Process {process.process_id} arrived; added to ready queue ",end="")
                print_queue(queue)
        time += 1
    time += int(t_cs/2)
    print(len(cpu_turnaround)+len(io_turnaround))
    print(f"time {time}ms: Simulator ended for RR ", end="")
    print_queue(queue)
    
    simout_write("RR",processes,time,cpu_turnaround,io_turnaround,cpu_wait,io_wait,cpu_context,io_context,cpu_preemp,io_preemp)

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
    
    simout = open("simout.txt", "w")

    print(f"\n<<< PROJECT PART II -- t_cs={t_cs}ms; alpha={alpha}; t_slice={t_slice}ms >>>")

    #first_come_first_serve(procs)
    #shortest_job_first(procs)
    shortest_remaining_time(procs)
    #round_robin(procs)