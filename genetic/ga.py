import numpy as np
import random
import os
import cProfile
import pstats
import time
from joblib import Parallel, delayed
from Graph import Graph

def path_length(distance_matrix, path):
    """Calculate total length of a path"""
    length = 0
    n = len(path)
    for i in range(n):
        a, b = path[i], path[(i + 1) % n]
        length += distance_matrix[a][b]
    return length

def process_batch(distance_matrix, batch):
    """Calculate lengths for a batch of paths"""
    return [(path, path_length(distance_matrix, path)) for path in batch]

class GeneticAlgorithmTSP:
    def __init__(self, graph, population_size=100, generations=500, 
                 mutation_rate=0.01, elite_size=5, n_jobs=-1):
        self.graph = graph
        self.distance_matrix = graph.distance_matrix
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.city_count = len(graph)
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()

    def _create_individual(self):
        """Create random path"""
        return np.random.permutation(self.city_count).tolist()

    def _initial_population(self):
        """Initialize population"""
        return [self._create_individual() for _ in range(self.population_size)]

    def _rank_population(self, population):
        """Evaluate and rank population using parallel processing"""
        if not population:
            return []
            
        # Ограничиваем число чанков размером популяции
        n_chunks = min(self.n_jobs * 4, len(population))
        chunks = self._chunkify(population, n_chunks)
        
        # Удаляем пустые чанки
        chunks = [ch for ch in chunks if ch]
        
        if not chunks:
            return []

        # Process batches in parallel
        results = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(process_batch)(self.distance_matrix, chunk)
            for chunk in chunks
        )
        
        # Flatten results and sort by fitness
        flat_results = [item for sublist in results for item in sublist]
        return sorted(flat_results, key=lambda x: x[1])

    def _chunkify(self, lst, n):
        """Split list into approximately equal chunks"""
        if n <= 0:
            return []
        n = min(n, len(lst))  # Не больше чем элементов в списке
        k, m = divmod(len(lst), n)
        chunks = []
        start = 0
        for i in range(n):
            end = start + k + (1 if i < m else 0)
            chunks.append(lst[start:end])
            start = end
        return chunks

    def _selection(self, ranked_population):
        """Select individuals for next generation"""
        if not ranked_population:
            return []
            
        # Elite selection
        selected = [individual for individual, _ in ranked_population[:self.elite_size]]
        
        # Fitness-proportional selection with protection
        fitnesses = []
        for _, length in ranked_population:
            if length <= 0:
                # Защита от нулевой длины
                fitnesses.append(1e10)
            else:
                fitnesses.append(1.0 / length)
        
        total_fitness = sum(fitnesses)
        
        # Если общий fitness нулевой, используем равномерное распределение
        if total_fitness <= 0:
            probabilities = None
        else:
            probabilities = [f / total_fitness for f in fitnesses]

        # Select remaining individuals
        selected.extend(
            random.choices(
                [ind for ind, _ in ranked_population],
                weights=probabilities,
                k=self.population_size - self.elite_size
            )
        )
        return selected

    def _crossover(self, parent1, parent2):
        """Ordered crossover (OX)"""
        start, end = sorted(random.sample(range(self.city_count), 2))
        child_p1 = parent1[start:end]
        child_p1_set = set(child_p1)
        child_p2 = [gene for gene in parent2 if gene not in child_p1_set]
        return child_p2[:start] + child_p1 + child_p2[start:]

    def _mutate(self, individual):
        """Swap mutation"""
        for i in range(self.city_count):
            if random.random() < self.mutation_rate:
                j = random.randint(0, self.city_count - 1)
                individual[i], individual[j] = individual[j], individual[i]
        return individual

    def _next_generation(self, current_pop):
        """Create new generation"""
        ranked = self._rank_population(current_pop)
        if not ranked:
            return self._initial_population()
            
        selected = self._selection(ranked)
        next_gen = []
        
        # Preserve elites
        next_gen.extend(selected[:self.elite_size])
        
        # Breed new individuals
        for _ in range(self.elite_size, self.population_size):
            parent1, parent2 = random.sample(selected, 2)
            child = self._mutate(self._crossover(parent1, parent2))
            next_gen.append(child)
            
        return next_gen

    def run(self):
        """Run genetic algorithm"""
        population = self._initial_population()
        best_path, best_length = None, float('inf')
        
        for generation in range(self.generations):
            population = self._next_generation(population)
            
            if not population:
                population = self._initial_population()
                print("Restarting population due to extinction")
                
            current_best = self._rank_population(population)[0] if population else (None, float('inf'))
            
            if current_best[1] < best_length:
                best_path, best_length = current_best[0], current_best[1]
            
            if generation % 15 == 0:
                print(f"Generation {generation+1}/{self.generations}: {current_best[1]:.2f}")
                
        return best_length, best_path

if __name__ == '__main__':
    # Example usage
    graph = Graph()
    graph.load("2d80.txt")  
    ga = GeneticAlgorithmTSP(
        graph,
        population_size=10000,
        generations=700,
        mutation_rate=0.01,
        elite_size=1000,
        n_jobs=6 
    )
    profiler = cProfile.Profile()
    profiler.enable()

    start_time = time.time()
    length, path = ga.run()
    duration = time.time() - start_time

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(40)  # Вывести топ 40 функций по времени

    print(f"\nOptimal path length: {length:.2f}")
    print(f"Path: {path}")
    print(f"Execution time: {duration:.2f} seconds") 
