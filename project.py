#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import math

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
    
    for process in range(n):
        
        num = next_exp(rand, lamb)
        
        while (num < 0) | (num > ceil):
            num = next_exp(rand, lamb)
            
        arrival_time = math.floor(num)
        bursts = math.ceil(rand.drand()*64)
        
        print(f"I/O-bound process {alph[process]}: arrival time {arrival_time}ms; {bursts} CPU bursts:")
        
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
                
            if burst >= bursts-1:
                terminal_out(round(cpu_burst_time))
            else:
                terminal_out(round(cpu_burst_time),round(io_burst_time))
    
def terminal_out(cpu_burst, io_burst=None):
    if not io_burst:
        print(f"--> CPU burst {cpu_burst}ms")
    else:
        print(f"--> CPU burst {cpu_burst}ms --> I/O burst {io_burst}ms")

if __name__ == '__main__':
    try:
        assert len(sys.argv) == 6
    except Exception as e:
        print("ERROR: Incorrect number of arguments", e, file=sys.stderr)
        sys.exit(1)
    try: 
        n = int(sys.argv[1])
        n_cpu = int(sys.argv[2])
        seed = int(sys.argv[3])
        lamb = float(sys.argv[4])
        ceil = int(sys.argv[5])
    except ValueError as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
    processes(n,n_cpu,seed,lamb,ceil)
            
    
