import os
import random
import time
import sys
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
from Graph import Graph

_doublepp = ndpointer(dtype=np.uintp, ndim=1, flags='C')

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, f'c_aco', f'libaco.so')
_external_ant_colony = ctypes.CDLL(file_path)

_run = _external_ant_colony.run
_run.argtypes = [
    _doublepp,                       # closeness_matrix
    _doublepp,                       # pheromone_matrix
    ctypes.c_size_t,                 # node_count
    ctypes.c_size_t,                 # ant_count
    ctypes.c_double,                 # A
    ctypes.c_double,                 # B
    ctypes.c_double,                 # Q
    ctypes.c_double,                 # evap
    ctypes.c_double,                 # start_ph
    ctypes.c_size_t,                 # k
    ctypes.POINTER(ctypes.c_double)  # best_len
]
_run.restype = ctypes.POINTER(ctypes.c_size_t)

_free_better_path = _external_ant_colony.free_better_path

class ACO:
    def __init__(self, graph):
        self.graph = graph

    def run(self, ant_count, A, B, Q, E, start_ph, k):
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
        start_ph = ctypes.c_double(start_ph)
        k = ctypes.c_size_t(k)
        best_len = ctypes.c_double()

        result_ptr = _run(dmpp, pmpp, node_count, ant_count, A, B, Q, E, start_ph, k, ctypes.byref(best_len))

        if result_ptr:
            best_path = np.ctypeslib.as_array(result_ptr, shape=(node_count.value,))
            return best_len.value, best_path
        else:
            return float("inf"), []
       
if __name__ == "__main__":
    """with open("logs.txt", "w") as file:
        graph = Graph()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'benchmarks', f'4d1000.txt')
        graph.load(file_path, ph=0.5)
        graph.add_k_nearest_edges(999)

        aco = ACO(graph)
        aco.run(1000, 3, 9, 10_000, 0.3, 0.5, 2_000)"""

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

    graph.add_k_nearest_edges(26)
    aco = ACO(graph)
    print(aco.run(200, 3, 9, 1000, 0.3, 0.5, 9))
