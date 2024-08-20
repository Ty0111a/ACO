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

_step = _external_ant_colony.step
_step.argtypes = [_doublepp, _doublepp, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
_step.restype = ctypes.POINTER(ctypes.c_size_t)

_free_better_path = _external_ant_colony.free_better_path

class ACO:
    def __init__(self, graph):
        self.graph = graph

    def step(self, ant_count, A, B, Q, E):
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
        bpl = ctypes.c_double()
        try:
            result_ptr = _step(dmpp, pmpp, node_count, ant_count, A, B, Q, E, ctypes.byref(bpl))
            if result_ptr:
                better_path = np.ctypeslib.as_array(result_ptr, shape=(1, node_count.value))[0]
            else:
                return float("inf"), []
        except Exception as e:
            print(f"{e}")
            return float("inf"), []
        finally:
            _external_ant_colony.free_better_path(result_ptr)
    
        return bpl.value, better_path

    def run_performance(self, ant_count, A, B, Q, evap, start_ph, worktime, fine):
        performances = []
        self.graph.setPH(ph=start_ph)
        # best_path_len = self.graph.lenRandomPath()
        # best_path_len = float("inf")
        best_path_len = fine
        startTime = time.time()
        while time.time() - startTime < worktime:
            bpl, _ = self.step(ant_count=ant_count, A=A, B=B, Q=Q, E=evap)
            if best_path_len > bpl:
                performances.append((time.time() - startTime, bpl))
                best_path_len = bpl

        performance = 0
        for i in range(1, len(performances)):
            performance += (performances[i][0] - performances[i - 1][0]) * performances[i][1]

        try:
            if performances[-1][0] > worktime:
                performance -= (performances[-1][0] - worktime) * performances[-1][1]
            else:
                performance += (worktime - performances[-1][0]) * performances[-1][1]
        except IndexError:
            pass

        performance += best_path_len * 3 * worktime
        return performance

    def run_print(self, ant_count, A, B, Q, evap, start_ph, worktime):
        print("let's go")
        # TODO: написать отображение графика эффективности
        self.graph.setPH(start_ph)
        # best_path_len = self.graph.lenRandomPath()
        best_path_len = float("inf")
        startTime = time.time()
        while time.time() - startTime < worktime:
            bpl, _ = self.step(ant_count=ant_count, A=A, B=B, Q=Q, E=evap)
            if best_path_len > bpl:
                print(f"{time.time() - startTime} {bpl}")
                best_path_len = bpl

    def run(self, ant_count, A, B, Q, evap, start_ph, worktime):
        self.graph.setPH(start_ph)
        best_path = []
        # best_path_len = self.graph.lenRandomPath()
        best_path_len = float("inf")
        startTime = time.time()
        while time.time() - startTime < worktime:
            bpl, bp = self.step(ant_count=ant_count, A=A, B=B, Q=Q, E=evap)
            if best_path_len > bpl:
                best_path_len = bpl
                best_path = bp
                print(f"{time.time() - startTime} {bpl}")

        print(best_path_len)
        return best_path



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
    print(aco.run(200, 3, 9, 1000, 0.3, 0.5, 2))
