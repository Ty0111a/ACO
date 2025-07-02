import ctypes
import time
import os
import numpy as np
from Graph import Graph
from numpy.ctypeslib import ndpointer

_doublepp = ndpointer(dtype=np.uintp, ndim=1, flags='C')

current_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(current_dir, "libbrute.so")
_lib = ctypes.CDLL(lib_path)

_run_brute = _lib.run_brute_force
_run_brute.argtypes = [
    _doublepp,                      # distance_matrix
    ctypes.c_size_t,                # node_count
    ctypes.POINTER(ctypes.c_double) # best_len
]
_run_brute.restype = ctypes.POINTER(ctypes.c_size_t)

_free_result = _lib.free_brute_result

class BruteForce:
    def __init__(self, graph):
        self.graph = graph

    def run(self):
        # Получаем указатели на строки матрицы расстояний
        dmpp = (self.graph.distance_matrix.__array_interface__['data'][0] +
                np.arange(self.graph.distance_matrix.shape[0]) * self.graph.distance_matrix.strides[0]
                ).astype(np.uintp)
        node_count = ctypes.c_size_t(self.graph.distance_matrix.shape[0])
        best_len = ctypes.c_double()

        result = _run_brute(dmpp, node_count, ctypes.byref(best_len))
        if result:
            path = [result[i] for i in range(node_count.value)]
            _free_result(result)
            return best_len.value, path
        else:
            return float("inf"), []


if __name__ == "__main__":
    graph = Graph()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    graph_name = f"2d14"
    file_path = os.path.join(current_dir, 'benchmarks', f'{graph_name}.txt')
    graph.load(file_path)
    bf = BruteForce(graph)
    st = time.time()
    length, path = bf.run()
    et = time.time()
    print("Best path:", path)
    print("Length:", length)
    print("Time: ", et - st)
