import os
import random
import statistics

from ACO import ACO, Graph, _init_rand

if __name__ == "__main__":
    _init_rand(random.randint(0, 4294967295))
    aco_worktime = 2
    graph = Graph()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'benchmarks', '2d100.txt')
    graph.load(file_path)
    graph.add_k_nearest_edges(99)
    aco = ACO(graph)
    parameters = {
        "ant_count": [100],
        "A": [0.25, 0.5, 1, 1.5, 2, 3, 4, 5],
        "B": [3, 4, 5, 6, 7, 8, 9, 10, 11],
        "Q": range(500, 10000, 500),
        "evap": [0.2],
        "start_ph": [0.5],
    }

    population_size = 128
    population = []

    for i in range(population_size):
        individual = {}
        for key in parameters:
            individual[key] = random.choice(parameters[key])
        population.append(individual)

    for l in range(20):
        print(f"let's go {l}")
        s = 0
        for individual in population:
            all_perfomance = [
                aco.run_performance(ant_count=int(individual["ant_count"]), A=individual["A"], B=individual["B"],
                                    Q=individual["Q"], evap=individual["evap"], start_ph=individual["start_ph"],
                                    worktime=aco_worktime) for _ in range(50)]
            individual["performance"] = statistics.mean(all_perfomance)
            s += individual["performance"]
            print(individual)

        print(f"avg {s // population_size}")

        print(min(population, key=lambda x: x["performance"]))

        # elitism
        elite_size = int(population_size * 0.3)
        elite = sorted(population, key=lambda x: x["performance"])[:elite_size]


        def crossover(mommy, daddy):
            child = {}
            for j in ["ant_count", "A", "B", "Q", "evap", "start_ph"]:
                minj = min(mommy[j], daddy[j])
                maxj = max(mommy[j], daddy[j])
                dmin = minj - 0.25 * (maxj - minj)
                dmax = minj + 0.25 * (maxj - minj)
                child[j] = random.uniform(dmin, dmax)

            for key in child:
                child[key] = round(abs(child[key]), 2)
            if child["evap"] > 0.99:
                child["evap"] = 0.99

            return child


        def mutate(child):
            for i in ["ant_count", "A", "B", "Q", "evap", "start_ph"]:
                if random.random() < 0.5:
                    child[i] = random.triangular(min(parameters[i]), max(parameters[i]),
                                                 random.gauss(child[i], child[i] * 0.2))

            for key in child:
                child[key] = abs(child[key])
            if child["evap"] > 0.99: child["evap"] = 0.99
            return child


        # crossing and mutate
        offspring = []

        for i in range(population_size - elite_size):
            parent1 = random.choice(elite)
            parent2 = random.choice(elite)
            child = crossover(parent1, parent2)
            child = mutate(child)
            offspring.append(child)

        population = elite + offspring
