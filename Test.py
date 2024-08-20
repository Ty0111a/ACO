from Graph import Graph
from ACO import ACO
import os



current_dir = os.path.dirname(os.path.abspath(__file__))
logsfile = os.path.join(current_dir, f"log.txt")

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'benchmarks', f'3d200.txt')

for k in range(200, 20, -1):    
    graph = Graph()
    graph.load(file_path, ph=0.5)
    graph.add_k_nearest_edges(k)
    aco = ACO(graph)
    res = [aco.run_performance(ant_count=200, A=3, B=9, Q=1_000, evap=0.30, start_ph=0.50, worktime=1,
                                fine=10_000) for _ in range(15)]
    with open(logsfile, "a") as file:
        print(f"{k} {res}")
        print(f"{k} {res}", file=file)
