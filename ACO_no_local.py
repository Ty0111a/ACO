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
file_path = os.path.join(current_dir, f'c_aco', f'libaco_standard.so')
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
    ctypes.c_size_t,                 # k (count of generations)
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
    ctypes.c_size_t,                 # max_generations (break if arrive this)
    ctypes.POINTER(ctypes.c_double)  # best_len
]
_run_until_stable_solution.restype = ctypes.POINTER(ctypes.c_size_t)


_free_better_path = _external_ant_colony.free_better_path

class ACO:
    def __init__(self, graph):
        self.graph = graph

    @logging
    @timeit
    def run(self, ant_count, A, B, Q, E, start_ph, k, delta=None, max_generations=0, **info):
        if ant_count <= 0:
            return float("inf"), []
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
        delta = ctypes.c_double(delta) if delta is not None else None
        max_generations = ctypes.c_size_t(max_generations)
        best_len = ctypes.c_double()

        try:
            if delta == None:
                result = _run_fixed_generation(dmpp, pmpp, node_count, ant_count, A, B, Q, E, k, ctypes.byref(best_len))
            else:
                result = _run_until_stable_solution(dmpp, pmpp, node_count, ant_count, A, B, Q, E, k, delta, 
                                                    max_generations, ctypes.byref(best_len))
            if result:
                result = result[:node_count.value]
            else:
                return float("inf"), []

        except Exception as e:
            print(f"{e}")
            return float("inf"), []

        return best_len.value, result
       
def main(n):
    graph = Graph()
    current_dir = os.path.dirname(os.path.abspath(__file__))

    graph_name = f"2d{n}"

    file_path = os.path.join(current_dir, 'benchmarks', f'{graph_name}.txt')
    graph.load(file_path, ph=0.5)

    graph.load(file_path)
    graph.add_k_nearest_edges(n)
    aco = ACO(graph)
    for _ in range(1):
        result = aco.run(ant_count=n, 
                  A=0.5, 
                  B=5.5,
                  Q=120, 
                  E=0.2, 
                  start_ph=0.4, 
                  k=int(1.3 * n + 200),
                  delta=None,
                  max_generations=500
                  )
        try:
            print(f"{result[0][0]} {result[1]}")
        except _:
            print(result)

if __name__ == "__main__":
    main(500)
