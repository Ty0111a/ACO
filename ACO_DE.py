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

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

_doublepp = ndpointer(dtype=np.uintp, ndim=1, flags='C')

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, f'c_aco', f'libaco_DE.so')
_external_ant_colony = ctypes.CDLL(file_path)

_ant_step = _external_ant_colony.ant_step 
_ant_step.argtypes = [
    _doublepp,                       # closeness_matrix
    _doublepp,                       # pheromone_matrix
    ctypes.c_size_t,                 # node_count
    ctypes.c_size_t,                 # ant_count
    ctypes.POINTER(ctypes.c_double), # A
    ctypes.POINTER(ctypes.c_double), # B
    ctypes.c_double,                 # Q
    ctypes.c_double,                 # evap
    ctypes.POINTER(ctypes.c_double),  # best_len
    ctypes.POINTER(ctypes.c_double)  # all lens
]
_ant_step.restype = ctypes.POINTER(ctypes.c_size_t)


_free_better_path = _external_ant_colony.free_better_path

class ACO:
    def __init__(self, graph):
        self.graph = graph

    def run(self, f, cr, ant_count, Q, E, start_ph, **info):
        if ant_count <= 0:
            return float("inf"), []
        self.graph.setPH(start_ph)
        dmpp = (self.graph.closeness_matrix.__array_interface__['data'][0] + np.arange(
            self.graph.closeness_matrix.shape[0]) * self.graph.closeness_matrix.strides[0]).astype(np.uintp)
        pmpp = (self.graph.pheromone_matrix.__array_interface__['data'][0] + np.arange(
            self.graph.pheromone_matrix.shape[0]) * self.graph.pheromone_matrix.strides[0]).astype(np.uintp)
        node_count = ctypes.c_size_t(self.graph.pheromone_matrix.shape[0])
        ant_count = ctypes.c_size_t(ant_count)

        A_min, A_max = 0.01, 10.0
        B_min, B_max = 0.01, 10.0
        N = 10 
        unique_ant_count = ant_count.value // N
        A_vals = np.random.uniform(A_min, A_max, size=ant_count.value // N)
        B_vals = np.random.uniform(A_min, A_max, size=ant_count.value // N)
        A = np.repeat(A_vals, N)
        B = np.repeat(B_vals, N)
        A = (ctypes.c_double * ant_count.value)(*A)
        B = (ctypes.c_double * ant_count.value)(*B)
        Q = ctypes.c_double(Q)
        E = ctypes.c_double(E)
        best_len = ctypes.c_double()
        all_lens = (ctypes.c_double * ant_count.value)()
        #print("A values:", [a for a in A])
        #print("B values:", [b for b in B])
        result = _ant_step(dmpp, pmpp, node_count, ant_count, A, B, Q, E, ctypes.byref(best_len), all_lens)
        #print(best_len.value)
        prev_lengths_array = [all_lens[i] for i in range(ant_count.value)]
        all_lengths = np.array([all_lens[i] for i in range(ant_count.value)])
        avg_lengths = np.mean(all_lengths.reshape(-1, N), axis=1)
        prev_avg_lengths = avg_lengths.copy()
        #print(f"All path lengths: {prev_lengths_array}")
        #print(f"avg len {prev_avg_lengths}")

        #print(" ")
        F = f 
        CR = cr
        DE_gens = 100 
        prev_A = A_vals.copy()
        prev_B = B_vals.copy()
        best_A, best_B, global_best_len = None, None, float("inf")
        for i in range(DE_gens):
            # мутация
            mutant_A = [prev_A[x]+F*(prev_A[y]-prev_A[z]) for x, y, z in (random.sample(range(len(prev_A)), 3) for _ in range(unique_ant_count))]
            mutant_B = [prev_B[x]+F*(prev_B[y]-prev_B[z]) for x, y, z in (random.sample(range(len(prev_B)), 3) for _ in range(unique_ant_count))]
            # скрещивание
            trial_A = [mutant_A[j] if random.random() < CR else prev_A[j] for j in range(unique_ant_count)]
            trial_B = [mutant_B[j] if random.random() < CR else prev_B[j] for j in range(unique_ant_count)]
            # проверка выхода за границы
            trial_A = [(a if A_min <= a <= A_max else (prev_A[i] + (A_min if a < A_min else A_max)) / 2) for i, a in enumerate(trial_A)]
            trial_B = [(b if B_min <= b <= B_max else (prev_B[i] + (B_min if b < B_min else B_max)) / 2) for i, b in enumerate(trial_B)]
            #print(f"Generation {i}:")
            #print("A values:", *[float(round(a, 2)) for a in trial_A])
            #print("B values:", *[float(round(b, 2)) for b in trial_B])
            # оценка приспособленности
            A = np.repeat(trial_A, N)
            B = np.repeat(trial_B, N)
            A = (ctypes.c_double * len(A))(*A)
            B = (ctypes.c_double * len(B))(*B)
            result = _ant_step(dmpp, pmpp, node_count, ant_count, A, B, Q, E, ctypes.byref(best_len), all_lens)
            lengths_array = [all_lens[i] for i in range(ant_count.value)]
            current_lengths = np.array([all_lens[i] for i in range(ant_count.value)])
            current_avg_lengths = np.mean(current_lengths.reshape(-1, N), axis=1)
            # Поиск параметров с минимальным маршрутом 
            lens_array = [all_lens[i] for i in range(ant_count.value)]
            min_index = int(np.argmin(lens_array))
            ant_group_index = min_index // N 
            best_A = trial_A[ant_group_index]
            best_B = trial_B[ant_group_index]
            global_best_len = lens_array[min_index]
            #print(best_len.value)
            # селекция (замещение) 
            for ant in range(unique_ant_count):
                if prev_avg_lengths[ant] > current_avg_lengths[ant]:
                    prev_A[ant] = float(trial_A[ant])
                    prev_B[ant] = float(trial_B[ant])
                    prev_avg_lengths[ant] = current_avg_lengths[ant]
            #print(f"All path lengths: {lengths_array}")
            #print(" ")
        for i in range(100 - DE_gens):
            #A = (ctypes.c_double * len(A))(*([0.6]*100))
            A = (ctypes.c_double * len(A))(*([best_A]*100))
            #B = (ctypes.c_double * len(B))(*([7.5]*100))
            B = (ctypes.c_double * len(B))(*([best_B]*100))
            result = _ant_step(dmpp, pmpp, node_count, ant_count, A, B, Q, E, ctypes.byref(best_len), all_lens)
            lengths_array = [all_lens[i] for i in range(ant_count.value)]
            current_lengths = np.array([all_lens[i] for i in range(ant_count.value)])
            current_avg_lengths = np.mean(current_lengths.reshape(-1, N), axis=1)
            #print(best_len.value)
        #print(best_A)
        #print(best_B)
        #print(global_best_len)


        return best_len.value, result
       
if __name__ == "__main__":
    graph = Graph()
    current_dir = os.path.dirname(os.path.abspath(__file__))

    graph_name = f"2d100"

    file_path = os.path.join(current_dir, 'benchmarks', f'{graph_name}.txt')
    graph.load(file_path, ph=0.4)

    graph.load(file_path)
    graph.add_k_nearest_edges(99)
    aco = ACO(graph)
    for f in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for cr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            result = [aco.run(ant_count=100, 
                  f=f,
                  cr=cr,
                  Q=120, 
                  E=0.2, 
                  start_ph=0.4)[0] for _ in range(10)]
            print(f"{cr} {f} {np.mean(result)}")

