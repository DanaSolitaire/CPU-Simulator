#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import numpy as np

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


if __name__ == '__main__':
    
    assert len(sys.argv) == 6, "Incorrect number of arguments"
    
    n = sys.argv[1]
    n_cpu = sys.argv[2]
    seed = sys.argv[3]
    lamb = sys.argv[4]
    ceil = sys.argv[5]
    
    
