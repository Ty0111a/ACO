import os
import random
import time
import sys
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
from Graph import Graph

from timeit import timeit
import line_profiler
from logger import logging

_doublepp = ndpointer(dtype=np.uintp, ndim=1, flags='C')

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, f'c_aco', f'libaco.so')
_external_ant_colony = ctypes.CDLL(file_path)

_run_fixed_generation = _external_ant_colony.run_fixed_generation
_run_fixed_generation.argtypes = [
    _doublepp,                       # closeness_matrix
    _doublepp,                       # pheromone_matrix
    ctypes.c_size_t,                 # node_count
    ctypes.c_size_t,                 # ant_count
    ctypes.c_double,                 # A
    ctypes.c_double,                 # B
    ctypes.c_double,                 # Q
    ctypes.c_double,                 # evap
    ctypes.c_size_t,                 # k
    ctypes.POINTER(ctypes.c_double)  # best_len
]
_run_fixed_generation.restype = ctypes.POINTER(ctypes.c_size_t)

_run_until_stable_solution = _external_ant_colony.run_until_stable_solution 
_run_until_stable_solution.argtypes = [
    _doublepp,                       # closeness_matrix
    _doublepp,                       # pheromone_matrix
    ctypes.c_size_t,                 # node_count
    ctypes.c_size_t,                 # ant_count
    ctypes.c_double,                 # A
    ctypes.c_double,                 # B
    ctypes.c_double,                 # Q
    ctypes.c_double,                 # evap
    ctypes.c_size_t,                 # k (repeated solution count)
    ctypes.c_double,                 # delta
    ctypes.POINTER(ctypes.c_double)  # best_len
]
_run_until_stable_solution.restype = ctypes.POINTER(ctypes.c_size_t)


_free_better_path = _external_ant_colony.free_better_path

class ACO:
    def __init__(self, graph):
        self.graph = graph

    @logging
    @line_profiler.profile
    def run(self, ant_count, A, B, Q, E, start_ph, k, delta=None, **info):
        self.graph.setPH(start_ph)
        dmpp = (self.graph.closeness_matrix.__array_interface__['data'][0] + np.arange(
            self.graph.closeness_matrix.shape[0]) * self.graph.closeness_matrix.strides[0]).astype(np.uintp)
        pmpp = (self.graph.pheromone_matrix.__array_interface__['data'][0] + np.arange(
            self.graph.pheromone_matrix.shape[0]) * self.graph.pheromone_matrix.strides[0]).astype(np.uintp)
        node_count = ctypes.c_size_t(self.graph.pheromone_matrix.shape[0])
        ant_count = ctypes.c_size_t(ant_count)
        A = ctypes.c_double(A)
        B = ctypes.c_double(B)
        Q = ctypes.c_double(Q)
        E = ctypes.c_double(E)
        k = ctypes.c_size_t(k)
        delta = ctypes.c_double(delta) if delta != None else None
        best_len = ctypes.c_double()

        try:
            if delta == None:
                result = _run_fixed_generation(dmpp, pmpp, node_count, ant_count, A, B, Q, E, k, ctypes.byref(best_len))
            else:
                result = _run_until_stable_solution(dmpp, pmpp, node_count, ant_count, A, B, Q, E, k, delta, ctypes.byref(best_len))
            if result:
                result = result[:node_count.value]
            else:
                return float("inf"), []
        except Exception as e:
            print(f"{e}")
            return float("inf"), []

        return best_len.value, result
       
if __name__ == "__main__":
    """with open("logs.txt", "w") as file:
        graph = Graph()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'benchmarks', f'4d1000.txt')
        graph.load(file_path, ph=0.5)
        graph.add_k_nearest_edges(999)

        aco = ACO(graph)
        aco.run_fixed_generation(1000, 3, 9, 10_000, 0.3, 0.5, 2_000)"""

    '''current_dir = os.path.dirname(os.path.abspath(__file__))
    logsfile = os.path.join(current_dir, f"log.txt")
    with open(logsfile, "w") as file:
        graph = Graph()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'benchmarks', f'4d1000.txt')
        graph.load(file_path, ph=0.5)

        for k in range(1000, 100, -50):
            graph.add_k_nearest_edges(k)

            aco = ACO(graph)
            res = [aco.run_performance(ant_count=1000,
                                       A=3,
                                       B=9,
                                       Q=10_000,
                                       evap=0.30,
                                       start_ph=0.50,
                                       worktime=600,
                                       fine=70_000) for _ in range(1)]

            print(f"{k} {res}")
            print(f"{k} {res}", file=file)'''

    graph = Graph()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'benchmarks', f'3d200.txt')
    graph.load(file_path, ph=0.5)

    graph.add_k_nearest_edges(199)
    aco = ACO(graph)
    print(aco.run(ant_count=100, 
                  A=3, 
                  B=9,
                  Q=1000, 
                  E=0.3, 
                  start_ph=0.5, 
                  k=20,
                  delta=0,
                  graph="3d200", nearest=199))
